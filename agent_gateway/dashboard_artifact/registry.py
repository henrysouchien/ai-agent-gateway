from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


StructuralValidator = Callable[[dict[str, Any]], list[str]]

ALLOWED_STATUSES = frozenset({"good", "watch", "risk", "bull", "base", "bear", "gap"})


@dataclass(frozen=True)
class ModuleSpec:
  name: str
  description: str
  required_keys: tuple[str, ...]
  material_keys: tuple[str, ...]
  structural_validator: StructuralValidator
  chart_minimums: dict[str, int] = field(default_factory=dict)

  def describe(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "description": self.description,
      "required_keys": list(self.required_keys),
      "material_keys": list(self.material_keys),
      "chart_minimums": dict(self.chart_minimums),
    }


def _validate_decision_box(data: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  if not _non_empty_string(data.get("stance")):
    errors.append("decision_box.stance must be a non-empty string")
  if not _non_empty_string(data.get("summary")):
    errors.append("decision_box.summary must be a non-empty string")
  _append_status_error(errors, data.get("status"), "decision_box.status")
  return errors


def _validate_metric_tiles(data: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  items = data.get("items")
  if not isinstance(items, list) or not items:
    return ["metric_tiles.items must be a non-empty list"]
  for index, item in enumerate(items):
    path = f"metric_tiles.items[{index}]"
    if not isinstance(item, dict):
      errors.append(f"{path} must be an object")
      continue
    if not _non_empty_string(item.get("label")):
      errors.append(f"{path}.label must be a non-empty string")
    if "value" not in item or item.get("value") in (None, ""):
      errors.append(f"{path}.value is required")
    if "detail" in item and item.get("detail") is not None and not isinstance(item.get("detail"), str):
      errors.append(f"{path}.detail must be a string when present")
    _append_status_error(errors, item.get("status"), f"{path}.status")
  return errors


def _validate_table(data: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  columns = data.get("columns")
  rows = data.get("rows")
  if not isinstance(columns, list) or not columns:
    errors.append("table.columns must be a non-empty list")
  elif not all(_non_empty_string(column) for column in columns):
    errors.append("table.columns must contain non-empty strings")
  if not isinstance(rows, list) or not rows:
    errors.append("table.rows must be a non-empty list")
  else:
    for index, row in enumerate(rows):
      if not isinstance(row, (dict, list)):
        errors.append(f"table.rows[{index}] must be an object or list")
      elif isinstance(row, list) and not row:
        errors.append(f"table.rows[{index}] must not be empty")
  return errors


def _validate_scenario_map(data: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  cases = data.get("cases")
  if not isinstance(cases, list) or not cases:
    return ["scenario_map.cases must be a non-empty list"]
  for index, case in enumerate(cases):
    path = f"scenario_map.cases[{index}]"
    if not isinstance(case, dict):
      errors.append(f"{path} must be an object")
      continue
    for key in ("label", "title", "body"):
      if not _non_empty_string(case.get(key)):
        errors.append(f"{path}.{key} must be a non-empty string")
    bullets = case.get("bullets")
    if bullets is not None and (
      not isinstance(bullets, list) or not all(_non_empty_string(item) for item in bullets)
    ):
      errors.append(f"{path}.bullets must contain non-empty strings when present")
    _append_status_error(errors, case.get("status"), f"{path}.status")
  return errors


def _validate_bar_chart(data: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  items = data.get("items")
  if not isinstance(items, list) or not items:
    return ["bar_chart.items must be a non-empty list"]
  for index, item in enumerate(items):
    path = f"bar_chart.items[{index}]"
    if not isinstance(item, dict):
      errors.append(f"{path} must be an object")
      continue
    if not _non_empty_string(item.get("label")):
      errors.append(f"{path}.label must be a non-empty string")
    if not _is_number(item.get("value")):
      errors.append(f"{path}.value must be numeric")
  return errors


def _validate_trend_chart(data: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  periods = data.get("periods")
  if not isinstance(periods, list) or not periods:
    return ["trend_chart.periods must be a non-empty list"]
  for index, period in enumerate(periods):
    path = f"trend_chart.periods[{index}]"
    if not isinstance(period, dict):
      errors.append(f"{path} must be an object")
      continue
    if not _non_empty_string(period.get("period")):
      errors.append(f"{path}.period must be a non-empty string")
    series_keys = [key for key, value in period.items() if key not in {"period", "citations"} and _is_number(value)]
    if not series_keys:
      errors.append(f"{path} must include at least one numeric series value")
  return errors


def _validate_timeline(data: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  events = data.get("events")
  if not isinstance(events, list) or not events:
    return ["timeline.events must be a non-empty list"]
  for index, event in enumerate(events):
    path = f"timeline.events[{index}]"
    if not isinstance(event, dict):
      errors.append(f"{path} must be an object")
      continue
    for key in ("time", "title"):
      if not _non_empty_string(event.get(key)):
        errors.append(f"{path}.{key} must be a non-empty string")
    if "detail" in event and event.get("detail") is not None and not isinstance(event.get("detail"), str):
      errors.append(f"{path}.detail must be a string when present")
  return errors


def _validate_text_block(data: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  if not _non_empty_string(data.get("body")):
    errors.append("text_block.body must be a non-empty string")
  bullets = data.get("bullets")
  if bullets is not None and (
    not isinstance(bullets, list) or not all(_non_empty_string(item) for item in bullets)
  ):
    errors.append("text_block.bullets must contain non-empty strings when present")
  return errors


def _validate_missing_evidence(data: dict[str, Any]) -> list[str]:
  items = data.get("items")
  if not isinstance(items, list) or not all(_non_empty_string(item) for item in items):
    return ["missing_evidence.items must be a list of strings"]
  return []


def chart_minimum_errors(module_type: str, data: dict[str, Any]) -> list[str]:
  if module_type == "bar_chart":
    items = data.get("items")
    if isinstance(items, list) and len(items) < 2:
      return ["bar_chart requires at least 2 items; drop the chart or add missing_evidence"]
  if module_type == "trend_chart":
    periods = data.get("periods")
    if not isinstance(periods, list):
      return []
    complete_periods = []
    for period in periods:
      if not isinstance(period, dict):
        continue
      has_series = any(key not in {"period", "citations"} and _is_number(value) for key, value in period.items())
      if _non_empty_string(period.get("period")) and has_series:
        complete_periods.append(period)
    if len(complete_periods) < 2:
      return ["trend_chart requires at least 2 complete periods; drop the chart or add missing_evidence"]
  return []


def _non_empty_string(value: object) -> bool:
  return isinstance(value, str) and bool(value.strip())


def _is_number(value: object) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _append_status_error(errors: list[str], value: object, path: str) -> None:
  if value is None:
    return
  if not isinstance(value, str) or value not in ALLOWED_STATUSES:
    errors.append(f"{path} must be one of {', '.join(sorted(ALLOWED_STATUSES))}")


MODULE_REGISTRY: dict[str, ModuleSpec] = {
  spec.name: spec
  for spec in (
    ModuleSpec(
      name="decision_box",
      description="Decision stance and concise evidence-backed summary.",
      required_keys=("stance", "summary"),
      material_keys=("summary",),
      structural_validator=_validate_decision_box,
    ),
    ModuleSpec(
      name="metric_tiles",
      description="Grid of labeled metric values with optional detail and status.",
      required_keys=("items",),
      material_keys=("items[].value", "items[].detail"),
      structural_validator=_validate_metric_tiles,
    ),
    ModuleSpec(
      name="table",
      description="Audit table with columns and row values.",
      required_keys=("columns", "rows"),
      material_keys=("rows[]",),
      structural_validator=_validate_table,
    ),
    ModuleSpec(
      name="scenario_map",
      description="Bull/base/bear or case cards with sourced bodies and bullets.",
      required_keys=("cases",),
      material_keys=("cases[].body", "cases[].bullets[]"),
      structural_validator=_validate_scenario_map,
    ),
    ModuleSpec(
      name="bar_chart",
      description="Ranked metric bar chart data for Recharts.",
      required_keys=("items",),
      material_keys=("items[].value",),
      structural_validator=_validate_bar_chart,
      chart_minimums={"min_items": 2},
    ),
    ModuleSpec(
      name="trend_chart",
      description="Period-indexed numeric series data for Recharts.",
      required_keys=("periods",),
      material_keys=("periods[].series_values",),
      structural_validator=_validate_trend_chart,
      chart_minimums={"min_complete_periods": 2},
    ),
    ModuleSpec(
      name="timeline",
      description="Catalyst or watch-item timeline.",
      required_keys=("events",),
      material_keys=("events[].detail",),
      structural_validator=_validate_timeline,
    ),
    ModuleSpec(
      name="text_block",
      description="Narrative prose block with optional bullets.",
      required_keys=("body",),
      material_keys=("body", "bullets[]"),
      structural_validator=_validate_text_block,
    ),
    ModuleSpec(
      name="missing_evidence",
      description="Explicit notice listing evidence gaps; citation-gap exempt.",
      required_keys=("items",),
      material_keys=(),
      structural_validator=_validate_missing_evidence,
    ),
  )
}


__all__ = [
  "ALLOWED_STATUSES",
  "MODULE_REGISTRY",
  "ModuleSpec",
  "chart_minimum_errors",
]
