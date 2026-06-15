from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit


DETAIL_MAX_CHARS = 80

_TEMPLATE_RE = re.compile(r"{([^{}]+)}")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _DisplaySpec:
  label: str
  detail_keys: tuple[str, ...] = ()


def _spec(label: str, *detail_keys: str) -> _DisplaySpec:
  return _DisplaySpec(label=label, detail_keys=tuple(detail_keys))


_SEED_MAP: dict[str, _DisplaySpec] = {
  # EDGAR / filings.
  "get_metric": _spec("Pulling {ticker} {metric_name}", "quarter+year", "role"),
  "get_metric_series": _spec("Pulling {ticker} {metric_name} history", "period", "role"),
  "get_financials": _spec("Pulling {ticker} financials", "statement", "period"),
  "get_filings": _spec("Finding {ticker} filings", "form", "period"),
  "get_filing_sections": _spec("Reading {ticker} filing sections", "form", "sections"),
  "get_filing_document": _spec("Reading {ticker} filing", "form", "sections"),
  "search_filing_text": _spec("Searching {ticker} filings", "query", "form"),
  "filings_search": _spec("Searching filings", "ticker", "query"),
  "filings_read": _spec("Reading filing excerpts", "ticker", "form"),
  # FMP / market data.
  "fmp_fetch": _spec("Pulling market data", "ticker", "statement", "period"),
  "fmp_profile": _spec("Pulling {ticker} profile", "symbol", "ticker"),
  "get_news": _spec("Pulling news", "ticker", "query"),
  "get_market_context": _spec("Pulling market context", "ticker", "query"),
  "get_economic_data": _spec("Pulling economic data", "symbol", "query"),
  "compare_peers": _spec("Comparing peers", "ticker", "peer_tickers"),
  "get_estimate_revisions": _spec("Pulling {ticker} estimate revisions", "period"),
  # Model / spreadsheet work.
  "build_model": _spec("Building a model", "ticker", "model"),
  "model_build": _spec("Building a model", "ticker", "model"),
  "model_forecast": _spec("Forecasting model values", "ticker", "scenario"),
  "model_find": _spec("Finding model items", "ticker", "query"),
  "model_values": _spec("Reading model values", "ticker", "range"),
  "model_sensitivity": _spec("Running sensitivity analysis", "ticker", "scenario"),
  "get_model_insights": _spec("Reviewing model insights", "ticker", "query"),
  "get_price_target": _spec("Pulling price target", "ticker", "scenario"),
  # Excel add-in.
  "read_cells": _spec("Reading cells", "workbook", "sheet", "range"),
  "write_cells": _spec("Writing cells", "workbook", "sheet", "range"),
  "list_sheets": _spec("Listing workbook sheets", "workbook"),
  "get_selection": _spec("Reading the current selection", "workbook", "sheet"),
  "get_used_range": _spec("Reading used range", "workbook", "sheet"),
  "apply_render_plan": _spec("Applying workbook updates", "sheet", "range"),
  "update_model": _spec("Updating the workbook model", "ticker", "sheet"),
  "find_cells": _spec("Finding cells", "query", "sheet"),
  # Local / orchestration / memory.
  "code_execute": _spec("Running a calculation"),
  "run_agent": _spec("Delegating to {agent}", "task"),
  "session_query": _spec("Searching the session", "query"),
  "memory_write": _spec("Writing {file}", "mode"),
  "memory_read": _spec("Reading {file}"),
  "memory_recall": _spec("Recalling memory", "query", "ticker"),
  "invoke_skill": _spec("Running {skill}", "ticker", "task"),
  # FMS / research and thesis workflow.
  "fms_persist_business_model": _spec("Saving business model work", "ticker"),
  "fms_persist_model_update": _spec("Saving model updates", "ticker"),
  "fms_propose_fundamental_research": _spec("Drafting fundamental research", "ticker"),
  "fms_propose_model_review": _spec("Drafting model review", "ticker"),
  "fms_propose_model_vs_consensus": _spec("Drafting model-vs-consensus view", "ticker"),
  "fms_propose_thesis_articulation": _spec("Drafting thesis articulation", "ticker"),
  "fms_report_build_model": _spec("Writing model-build report", "ticker"),
  "fms_report_business_quality_assessment": _spec("Writing business-quality report", "ticker"),
  "fms_report_idea_to_thesis": _spec("Writing idea-to-thesis report", "ticker"),
  "fms_report_thesis_consultation": _spec("Writing thesis-consultation report", "ticker"),
  "fms_report_risk_review": _spec("Writing risk review", "ticker"),
  "fms_link_thesis": _spec("Linking thesis records", "ticker", "name"),
}

_ALIASES: dict[str, tuple[str, ...]] = {
  "agent": ("agent", "agent_name", "skill", "skill_name", "name"),
  "command": ("command", "cmd"),
  "file": ("file", "path", "file_path", "filename", "artifact_path", "source_path"),
  "form": ("form", "form_type", "filing_type", "source"),
  "metric_name": ("metric_name", "metric", "concept", "tag"),
  "model": ("model", "model_name", "model_id", "view_model_id"),
  "mode": ("mode", "mutation_mode", "write_mode"),
  "name": ("name", "title"),
  "peer_tickers": ("peer_tickers", "peers", "peer_symbols"),
  "period": ("period", "fiscal_period", "date", "as_of"),
  "query": ("query", "search", "search_text", "prompt"),
  "range": ("range", "address", "cell_range", "target_range", "source_range"),
  "role": ("role", "statement", "statement_type", "section"),
  "scenario": ("scenario", "scenario_name", "case"),
  "sections": ("sections", "section", "items", "item"),
  "sheet": ("sheet", "sheet_name", "worksheet"),
  "skill": ("skill", "skill_name", "workflow_name"),
  "statement": ("statement", "statement_type", "role"),
  "symbol": ("symbol", "ticker"),
  "task": ("task", "prompt", "message", "description"),
  "task_id": ("task_id", "background_task_id", "run_id"),
  "ticker": ("ticker", "symbol", "company_ticker"),
  "url": ("url", "uri", "href"),
  "workbook": ("workbook", "workbook_name", "_workbook"),
}

_GLOBAL_SALIENT_KEYS: tuple[str, ...] = (
  "ticker",
  "query",
  "file",
  "range",
  "sheet",
  "metric_name",
  "symbol",
  "url",
  "name",
)


def resolve_display(tool_name: str, redacted_input: Mapping[str, Any] | None) -> dict[str, str] | None:
  """Return a product-facing display label for a redacted tool input."""
  canonical = _canonical_tool_name(tool_name)
  if not canonical:
    return None

  data = redacted_input if isinstance(redacted_input, Mapping) else {}
  spec = _SEED_MAP.get(canonical)
  if spec is not None:
    label = _clean_text(_render_template(spec.label, data))
    detail = _build_detail(data, spec.detail_keys)
  else:
    label = _humanize_tool_name(canonical)
    detail = _first_salient_detail(data)

  if not label:
    return None
  display = {"label": label}
  if detail:
    display["detail"] = detail
  return display


def _canonical_tool_name(tool_name: str) -> str:
  value = str(tool_name or "").strip()
  if value.startswith("mcp__"):
    parts = value.split("__", 2)
    if len(parts) == 3:
      value = parts[2]
  return value.strip()


def _humanize_tool_name(tool_name: str) -> str:
  leaf = tool_name.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
  words = re.sub(r"[-_]+", " ", leaf).strip()
  if not words:
    return ""
  words = words.lower()
  return words[:1].upper() + words[1:]


def _render_template(template: str, data: Mapping[str, Any]) -> str:
  def replace(match: re.Match[str]) -> str:
    return _lookup_text(data, match.group(1)) or ""

  return _TEMPLATE_RE.sub(replace, template)


def _build_detail(data: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
  parts: list[str] = []
  seen: set[str] = set()
  for key in keys:
    value = _lookup_text(data, key)
    if not value:
      continue
    normalized = value.casefold()
    if normalized in seen:
      continue
    seen.add(normalized)
    parts.append(value)
  if not parts:
    return None
  return _truncate_detail(" - ".join(parts))


def _first_salient_detail(data: Mapping[str, Any]) -> str | None:
  for key in _GLOBAL_SALIENT_KEYS:
    value = _lookup_text(data, key)
    if value:
      return _truncate_detail(value)
  return None


def _lookup_text(data: Mapping[str, Any], key: str) -> str | None:
  if key == "quarter+year":
    return _format_quarter_year(data)
  if key == "_workbook":
    return _lookup_text(data, "workbook")
  aliases = _ALIASES.get(key, (key,))
  for alias in aliases:
    if alias not in data:
      continue
    value = _format_value(alias, data.get(alias))
    if value:
      return value
  return None


def _format_quarter_year(data: Mapping[str, Any]) -> str | None:
  quarter = _lookup_raw(data, ("quarter", "fiscal_quarter"))
  year = _lookup_raw(data, ("year", "fiscal_year"))
  if quarter is None and year is None:
    return _lookup_text(data, "period")
  quarter_text = _format_value("quarter", quarter) if quarter is not None else None
  year_text = _format_value("year", year) if year is not None else None
  if quarter_text and quarter_text.isdigit():
    quarter_text = f"Q{quarter_text}"
  parts = [part for part in (quarter_text, year_text) if part]
  return " ".join(parts) if parts else None


def _lookup_raw(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
  for key in keys:
    if key in data and data.get(key) is not None:
      return data.get(key)
  return None


def _format_value(key: str, value: Any) -> str | None:
  if value is None:
    return None
  if key in {"url", "uri", "href"}:
    return _format_url(value)
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, (int, float)):
    text = str(value)
  elif isinstance(value, str):
    text = value.strip()
  elif isinstance(value, (list, tuple)):
    parts = [part for item in value[:3] if (part := _format_value(key, item))]
    text = ", ".join(parts)
    if len(value) > 3 and text:
      text = f"{text}, ..."
  elif isinstance(value, Mapping):
    text = ""
    for nested_key in ("name", "label", "title", "id"):
      nested_text = _format_value(nested_key, value.get(nested_key))
      if nested_text:
        text = nested_text
        break
  else:
    text = str(value).strip()
  text = _clean_text(text)
  if key in {"ticker", "symbol", "company_ticker"}:
    text = text.upper()
  return text or None


def _format_url(value: Any) -> str | None:
  raw = str(value or "").strip()
  if not raw:
    return None
  parsed = urlsplit(raw)
  if not parsed.scheme or not parsed.netloc:
    path = raw.split("?", 1)[0].split("#", 1)[0]
    return _truncate_detail(path)

  origin = f"{parsed.scheme}://{parsed.netloc}"
  path = parsed.path or ""
  if not path or path == "/":
    return origin
  max_path = max(0, DETAIL_MAX_CHARS - len(origin))
  if max_path <= 4:
    return _truncate_detail(origin)
  if len(path) > max_path:
    path = path[: max_path - 3].rstrip("/") + "..."
  return _truncate_detail(f"{origin}{path}")


def _clean_text(value: str) -> str:
  text = _WHITESPACE_RE.sub(" ", value).strip()
  text = re.sub(r"\s+([,.;:!?])", r"\1", text)
  return text


def _truncate_detail(value: str) -> str:
  text = _clean_text(value)
  if len(text) <= DETAIL_MAX_CHARS:
    return text
  return text[: DETAIL_MAX_CHARS - 3].rstrip() + "..."


__all__ = ["DETAIL_MAX_CHARS", "resolve_display"]
