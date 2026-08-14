from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict

from pydantic import ValidationError


def capture_filing_source_pack(
  source_pack_session_target: Any,
  result: Any,
  tool_input: Dict[str, Any],
  logger: Any,
  *,
  planner_result_payload_fn: Callable[[Any], Any | None] | None = None,
  derive_fiscal_period_fn: Callable[[Dict[str, Any], Any], str | None] | None = None,
  payload_get_fn: Callable[[Any, str], Any] | None = None,
) -> None:
  if source_pack_session_target is None:
    return
  try:
    from agent.shared import source_pack_session
    from schema.source_pack import SourcePack

    planner_result_payload_fn = planner_result_payload_fn or planner_result_payload
    derive_fiscal_period_fn = derive_fiscal_period_fn or derive_fiscal_period
    payload_get_fn = payload_get_fn or payload_get
    planner_result = planner_result_payload_fn(result)
    if planner_result is None:
      return
    pack = SourcePack.from_planner_result(
      planner_result,
      ticker=tool_input.get("ticker"),
      fiscal_period=derive_fiscal_period_fn(tool_input, planner_result),
      form_type=tool_input.get("form_type") or payload_get_fn(planner_result, "form_type"),
    )
    source_pack_session.store(source_pack_session_target, pack)
  except (ValidationError, TypeError, AttributeError, ValueError) as exc:
    logger.warning("get_filing_evidence result didn't adapt to SourcePack: %s", exc)


def planner_result_payload(result: Any) -> Any | None:
  return planner_result_payload_with_hooks(
    result,
    candidates_fn=planner_result_candidates,
    looks_like_fn=looks_like_source_pack_payload,
    coerce_fn=coerce_planner_result_payload,
  )


def planner_result_payload_with_hooks(
  result: Any,
  *,
  candidates_fn: Callable[[Any], list[Any]],
  looks_like_fn: Callable[[Any], bool],
  coerce_fn: Callable[[Any], Any],
) -> Any | None:
  for candidate in candidates_fn(result):
    if looks_like_fn(candidate):
      return coerce_fn(candidate)
  return None


def planner_result_candidates(result: Any) -> list[Any]:
  candidates: list[Any] = []
  for key in ("source_pack", "planner_result", "planner_trace", "result"):
    if isinstance(result, dict):
      candidates.append(result.get(key))
    else:
      candidates.append(getattr(result, key, None))
  candidates.append(result)
  return candidates


def coerce_planner_result_payload(payload: Any) -> Any:
  if isinstance(payload, dict):
    return SimpleNamespace(**payload)
  return payload


def looks_like_source_pack_payload(payload: Any) -> bool:
  required = ("matched_intent", "required_reads", "rationale")
  if isinstance(payload, dict):
    return all(key in payload for key in required)
  if payload is None:
    return False
  return all(hasattr(payload, key) for key in required)


def payload_get(payload: Any, key: str) -> Any:
  if isinstance(payload, dict):
    return payload.get(key)
  return getattr(payload, key, None)


def derive_fiscal_period(
  tool_input: Dict[str, Any],
  planner_result: Any,
  *,
  payload_get_fn: Callable[[Any, str], Any] | None = None,
) -> str | None:
  payload_get_fn = payload_get_fn or payload_get
  for key in ("fiscal_period", "period"):
    value = tool_input.get(key) or payload_get_fn(planner_result, key)
    if value:
      return str(value)
  year = tool_input.get("year") or tool_input.get("fiscal_year") or payload_get_fn(planner_result, "year")
  quarter = tool_input.get("quarter") or tool_input.get("fiscal_quarter") or payload_get_fn(planner_result, "quarter")
  if year and quarter:
    quarter_text = str(quarter).upper()
    if not quarter_text.startswith("Q"):
      quarter_text = f"Q{quarter_text}"
    return f"FY{year} {quarter_text}"
  if year:
    return f"FY{year}"
  return None


__all__ = [
  "capture_filing_source_pack",
  "coerce_planner_result_payload",
  "derive_fiscal_period",
  "looks_like_source_pack_payload",
  "payload_get",
  "planner_result_candidates",
  "planner_result_payload",
  "planner_result_payload_with_hooks",
]
