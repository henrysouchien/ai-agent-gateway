from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from pydantic import ValidationError

from schema.dashboard_payload import DashboardPayload

from .registry import MODULE_REGISTRY, chart_minimum_errors


VALIDATION_PROFILES = frozenset({"draft", "production"})
PRODUCTION_POSTURES = frozenset({"pm_ready", "decision_ready"})
NON_PRODUCTION_POSTURES = frozenset({"draft", "working_draft", "screen_grade"})

_SOURCE_MARKER_RE = re.compile(r"\[(src_[1-9]\d*)\]")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\$?\d[\d,]*(?:\.\d+)?%?)(?![A-Za-z])")
_SOURCE_ID_RE = re.compile(r"^src_[1-9]\d*$")
_PLACEHOLDER_PATTERNS = (
  re.compile(r"TODO", re.IGNORECASE),
  re.compile(r"FIXME", re.IGNORECASE),
  re.compile(r"\[[A-Z][A-Z0-9_ -]*\]"),
  re.compile(r"\{[A-Z][A-Z0-9_ -]*\}"),
  re.compile(r"undefined", re.IGNORECASE),
)


@dataclass(frozen=True)
class _MaterialValue:
  path: str
  value: Any
  citations: tuple[str, ...]


def validate_dashboard_payload(payload: Any, profile: str) -> dict[str, Any]:
  warnings: list[str] = []
  hard_failures: list[str] = []

  if profile not in VALIDATION_PROFILES:
    return _report(
      profile=profile,
      production_required=False,
      citation_strict=False,
      hard_failures=[f"profile must be one of {', '.join(sorted(VALIDATION_PROFILES))}"],
      warnings=[],
    )

  try:
    model = DashboardPayload.model_validate(payload)
  except ValidationError as exc:
    return _report(
      profile=profile,
      production_required=False,
      citation_strict=False,
      hard_failures=_schema_errors(exc),
      warnings=[],
    )

  source_ids = {source.id for source in model.sources}
  production_required = profile == "production" or model.metadata.readiness_posture in PRODUCTION_POSTURES
  citation_strict = model.metadata.citation_policy == "strict" or production_required
  payload_json = model.model_dump(mode="json")

  hard_failures.extend(_placeholder_errors(payload_json))
  hard_failures.extend(_module_errors(model))
  hard_failures.extend(_citation_reference_errors(payload_json, source_ids, citation_strict, warnings))
  _strict_or_warn(
    _unresolved_marker_errors(payload_json, source_ids),
    citation_strict=citation_strict,
    hard_failures=hard_failures,
    warnings=warnings,
  )
  _strict_or_warn(
    _citation_gap_errors(model),
    citation_strict=citation_strict,
    hard_failures=hard_failures,
    warnings=warnings,
  )
  if not model.sources:
    _strict_or_warn(
      ["source ledger must not be empty in citation-strict validation"],
      citation_strict=citation_strict,
      hard_failures=hard_failures,
      warnings=warnings,
    )

  freeze_time = _parse_iso(model.metadata.freeze_time)
  if freeze_time is None:
    _strict_or_warn(
      [f"metadata.freeze_time must parse as ISO-8601: {model.metadata.freeze_time!r}"],
      citation_strict=citation_strict,
      hard_failures=hard_failures,
      warnings=warnings,
    )
  else:
    hard_failures.extend(_freshness_errors(model, freeze_time, citation_strict, warnings, production_required))

  if production_required:
    hard_failures.extend(_production_errors(model))

  return _report(
    profile=profile,
    production_required=production_required,
    citation_strict=citation_strict,
    hard_failures=hard_failures,
    warnings=warnings,
  )


def build_dashboard_artifact(payload: Any, profile: str, summary: str) -> dict[str, Any]:
  summary_text = str(summary or "").strip()
  validation = validate_dashboard_payload(payload, profile)
  hard_failures = list(validation["hard_failures"])
  if not summary_text:
    hard_failures.append("summary must be a non-empty string")
  if hard_failures:
    return {
      "error": "dashboard_validation_failed",
      "hard_failures": hard_failures,
      "warnings": list(validation["warnings"]),
    }

  model = DashboardPayload.model_validate(payload)
  return {
    "payload_json": model.model_dump(mode="json"),
    "sidecar_fields": {
      "title": model.title,
      "summary": summary_text,
      "ticker": model.ticker,
      "scope_label": model.scope_label,
      "readiness_posture": model.metadata.readiness_posture,
      "profile": profile,
    },
    "warnings": list(validation["warnings"]),
  }


def _schema_errors(exc: ValidationError) -> list[str]:
  errors: list[str] = []
  for item in exc.errors():
    loc = ".".join(str(part) for part in item.get("loc", ())) or "payload"
    errors.append(f"payload schema validation failed at {loc}: {item.get('msg')}")
  return errors


def _module_errors(model: DashboardPayload) -> list[str]:
  errors: list[str] = []
  for section_index, section in enumerate(model.sections):
    for module_index, module in enumerate(section.modules):
      path = f"sections[{section_index}].modules[{module_index}]"
      spec = MODULE_REGISTRY.get(module.type)
      if spec is None:
        errors.append(f"{path}.type is not registered: {module.type}")
        continue
      for key in spec.required_keys:
        if key not in module.data:
          errors.append(f"{path}.data missing required key: {key}")
      errors.extend(f"{path}.data {error}" for error in spec.structural_validator(module.data))
  return errors


def _citation_gap_errors(model: DashboardPayload) -> list[str]:
  errors: list[str] = []
  for section_index, section in enumerate(model.sections):
    for module_index, module in enumerate(section.modules):
      spec = MODULE_REGISTRY.get(module.type)
      if spec is None:
        continue
      for error in chart_minimum_errors(module.type, module.data):
        errors.append(f"sections[{section_index}].modules[{module_index}].data {error}")
      for material in _iter_material_values(module.type, module.data):
        if _material_needs_citation(material.value) and not _material_is_cited(material):
          errors.append(
            f"citation gap at sections[{section_index}].modules[{module_index}].data.{material.path}"
          )
  return errors


def _iter_material_values(module_type: str, data: dict[str, Any]) -> Iterable[_MaterialValue]:
  module_citations = _citations(data)
  if module_type == "decision_box":
    yield _MaterialValue("summary", data.get("summary"), module_citations)
  elif module_type == "metric_tiles":
    for index, item in enumerate(_list_of_dicts(data.get("items"))):
      item_citations = _citations(item) or module_citations
      yield _MaterialValue(f"items[{index}].value", item.get("value"), item_citations)
      if "detail" in item:
        yield _MaterialValue(f"items[{index}].detail", item.get("detail"), item_citations)
  elif module_type == "table":
    for row_index, row in enumerate(data.get("rows") if isinstance(data.get("rows"), list) else []):
      row_citations = _citations(row) or module_citations if isinstance(row, dict) else module_citations
      if isinstance(row, dict):
        for key, value in row.items():
          if key == "citations":
            continue
          yield _MaterialValue(f"rows[{row_index}].{key}", value, row_citations)
      elif isinstance(row, list):
        for column_index, value in enumerate(row):
          yield _MaterialValue(f"rows[{row_index}][{column_index}]", value, row_citations)
  elif module_type == "scenario_map":
    for index, case in enumerate(_list_of_dicts(data.get("cases"))):
      case_citations = _citations(case) or module_citations
      yield _MaterialValue(f"cases[{index}].body", case.get("body"), case_citations)
      bullets = case.get("bullets")
      if isinstance(bullets, list):
        for bullet_index, bullet in enumerate(bullets):
          yield _MaterialValue(f"cases[{index}].bullets[{bullet_index}]", bullet, case_citations)
  elif module_type == "bar_chart":
    for index, item in enumerate(_list_of_dicts(data.get("items"))):
      item_citations = _citations(item) or module_citations
      yield _MaterialValue(f"items[{index}].value", item.get("value"), item_citations)
  elif module_type == "trend_chart":
    periods = data.get("periods")
    if isinstance(periods, list):
      for index, period in enumerate(periods):
        if not isinstance(period, dict):
          continue
        period_citations = _citations(period) or module_citations
        for key, value in period.items():
          if key in {"period", "citations"}:
            continue
          yield _MaterialValue(f"periods[{index}].{key}", value, period_citations)
  elif module_type == "timeline":
    for index, event in enumerate(_list_of_dicts(data.get("events"))):
      event_citations = _citations(event) or module_citations
      yield _MaterialValue(f"events[{index}].detail", event.get("detail"), event_citations)
  elif module_type == "text_block":
    yield _MaterialValue("body", data.get("body"), module_citations)
    bullets = data.get("bullets")
    if isinstance(bullets, list):
      for bullet_index, bullet in enumerate(bullets):
        yield _MaterialValue(f"bullets[{bullet_index}]", bullet, module_citations)


def _material_needs_citation(value: Any) -> bool:
  if isinstance(value, bool):
    return False
  if isinstance(value, (int, float)):
    return True
  return isinstance(value, str) and bool(_NUMBER_RE.search(value))


def _material_is_cited(material: _MaterialValue) -> bool:
  if isinstance(material.value, str) and _SOURCE_MARKER_RE.search(material.value):
    return True
  return bool(material.citations)


def _citation_reference_errors(
  value: Any,
  source_ids: set[str],
  citation_strict: bool,
  warnings: list[str],
) -> list[str]:
  issues: list[str] = []
  for path, refs in _iter_citation_lists(value):
    for ref in refs:
      if not isinstance(ref, str) or not _SOURCE_ID_RE.fullmatch(ref):
        issues.append(f"citation reference at {path} is not a valid src_N id: {ref!r}")
      elif ref not in source_ids:
        issues.append(f"citation reference at {path} does not resolve: {ref}")
  if citation_strict:
    return issues
  warnings.extend(issues)
  return []


def _iter_citation_lists(value: Any, path: str = "payload") -> Iterable[tuple[str, list[Any]]]:
  if isinstance(value, dict):
    for key, child in value.items():
      child_path = f"{path}.{key}"
      if key == "citations" and isinstance(child, list):
        yield child_path, child
      else:
        yield from _iter_citation_lists(child, child_path)
  elif isinstance(value, list):
    for index, child in enumerate(value):
      yield from _iter_citation_lists(child, f"{path}[{index}]")


def _unresolved_marker_errors(value: Any, source_ids: set[str]) -> list[str]:
  errors: list[str] = []
  for path, text in _iter_strings(value):
    for marker in _SOURCE_MARKER_RE.findall(text):
      if marker not in source_ids:
        errors.append(f"unresolved source marker {marker} at {path}")
  return errors


def _placeholder_errors(value: Any) -> list[str]:
  errors: list[str] = []
  for path, text in _iter_strings(value):
    for pattern in _PLACEHOLDER_PATTERNS:
      match = pattern.search(text)
      if match:
        errors.append(f"placeholder text {match.group(0)!r} at {path}")
        break
  return errors


def _iter_strings(value: Any, path: str = "payload") -> Iterable[tuple[str, str]]:
  if isinstance(value, str):
    yield path, value
  elif isinstance(value, dict):
    for key, child in value.items():
      yield from _iter_strings(child, f"{path}.{key}")
  elif isinstance(value, list):
    for index, child in enumerate(value):
      yield from _iter_strings(child, f"{path}[{index}]")


def _freshness_errors(
  model: DashboardPayload,
  freeze_time: datetime,
  citation_strict: bool,
  warnings: list[str],
  production_required: bool,
) -> list[str]:
  errors: list[str] = []
  for source in model.sources:
    if not source.retrieved_at:
      if production_required:
        warnings.append(f"source {source.id} has no retrieved_at; freshness unchecked")
      continue
    retrieved_at = _parse_iso(source.retrieved_at)
    if retrieved_at is None:
      if production_required:
        warnings.append(f"source {source.id} retrieved_at is not ISO-8601; freshness unchecked")
      continue
    if retrieved_at > freeze_time:
      issue = f"source {source.id} retrieved_at postdates metadata.freeze_time"
      if citation_strict:
        errors.append(issue)
      else:
        warnings.append(issue)
  return errors


def _production_errors(model: DashboardPayload) -> list[str]:
  errors: list[str] = []
  if not model.metadata.freeze_time.strip():
    errors.append("production dashboards require metadata.freeze_time")
  if not (model.metadata.decision_context or "").strip():
    errors.append("production dashboards require metadata.decision_context")
  if model.metadata.citation_policy != "strict":
    errors.append("production dashboards require metadata.citation_policy='strict'")
  if model.metadata.readiness_posture in NON_PRODUCTION_POSTURES:
    errors.append("production dashboards require readiness_posture pm_ready or decision_ready")
  if model.hero is None:
    errors.append("production dashboards require non-empty hero")
  if not model.snapshot:
    errors.append("production dashboards require non-empty snapshot")
  return errors


def _strict_or_warn(
  issues: list[str],
  *,
  citation_strict: bool,
  hard_failures: list[str],
  warnings: list[str],
) -> None:
  if citation_strict:
    hard_failures.extend(issues)
  else:
    warnings.extend(issues)


def _parse_iso(value: str) -> datetime | None:
  try:
    normalized = value.strip()
    if normalized.endswith("Z"):
      normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
  except (TypeError, ValueError):
    return None
  if parsed.tzinfo is None:
    return parsed.replace(tzinfo=timezone.utc)
  return parsed.astimezone(timezone.utc)


def _citations(value: Any) -> tuple[str, ...]:
  if not isinstance(value, dict):
    return ()
  citations = value.get("citations")
  if not isinstance(citations, list):
    return ()
  return tuple(str(item) for item in citations if str(item).strip())


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
  if not isinstance(value, list):
    return []
  return [item for item in value if isinstance(item, dict)]


def _report(
  *,
  profile: str,
  production_required: bool,
  citation_strict: bool,
  hard_failures: list[str],
  warnings: list[str],
) -> dict[str, Any]:
  return {
    "status": "fail" if hard_failures else "pass",
    "profile": profile,
    "production_required": production_required,
    "citation_strict": citation_strict,
    "hard_failures": hard_failures,
    "warnings": warnings,
  }


__all__ = [
  "build_dashboard_artifact",
  "validate_dashboard_payload",
]
