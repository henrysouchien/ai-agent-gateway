from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from agent_gateway.artifact_paths import canonicalize_ticker


CORPUS_READINESS_TOOL = "check_corpus_readiness"
CORPUS_REQUIREMENTS_FIELD = "corpus_requirements"
CORPUS_GATED_PIPELINE_TEMPLATES = frozenset({"valuation-ready"})

_PERIOD_RE = re.compile(r"^[0-9]{4}-(?:Q[1-4]|FY)$")
_REQUIREMENT_KEYS = frozenset({
  "ticker",
  "required_filings",
  "required_transcripts",
})
log = logging.getLogger("agent_gateway.control_plane.corpus_readiness")


class CorpusReadinessGateError(RuntimeError):
  def __init__(
    self,
    status_code: int,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
  ) -> None:
    super().__init__(message)
    self.status_code = int(status_code)
    self.code = str(code)
    self.message = str(message)
    self.details = dict(details or {})

  def to_payload(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "code": self.code,
      "message": self.message,
    }
    if self.details:
      payload["details"] = dict(self.details)
    return payload


async def require_corpus_readiness(
  payload: dict[str, Any],
  *,
  app_state: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  """Fail closed on declared corpus gaps before batch admission or spend."""

  normalized_payload, requirements = _normalize_requirements(payload)
  if not requirements:
    return normalized_payload, None

  gateway_config = getattr(app_state, "gateway_config", None)
  mcp_client = getattr(gateway_config, "mcp_client", None)
  if mcp_client is None or not callable(getattr(mcp_client, "call_tool", None)):
    raise _unavailable_error("The corpus readiness service is not configured.")

  get_server_for_tool = getattr(mcp_client, "get_server_for_tool", None)
  if callable(get_server_for_tool) and get_server_for_tool(CORPUS_READINESS_TOOL) is None:
    raise _unavailable_error("The corpus readiness tool is not connected.")

  checks = await asyncio.gather(
    *(
      _check_one_requirement(mcp_client, requirement)
      for requirement in requirements
    )
  )
  return normalized_payload, {
    "status": "ready",
    "checks": checks,
  }


async def _check_one_requirement(
  mcp_client: Any,
  requirement: dict[str, Any],
) -> dict[str, Any]:
  ticker = str(requirement["ticker"])
  tool_input = {
    "ticker": ticker,
    "required_filings": list(requirement["required_filings"]),
    "required_transcripts": list(requirement["required_transcripts"]),
  }
  try:
    result, error = await mcp_client.call_tool(CORPUS_READINESS_TOOL, tool_input)
  except asyncio.CancelledError:
    raise
  except Exception as exc:
    log.exception("Corpus readiness call failed for %s", ticker, exc_info=exc)
    raise _unavailable_error(
      "The corpus readiness service could not complete the check.",
      ticker=ticker,
    ) from exc

  if error is not None:
    log.error("Corpus readiness transport error for %s: %s", ticker, error)
    raise _unavailable_error(
      "The corpus readiness service could not complete the check.",
      ticker=ticker,
      upstream_code=str(error.get("code") or "mcp_error") if isinstance(error, dict) else "mcp_error",
    )
  if not isinstance(result, dict):
    raise _unavailable_error(
      "The corpus readiness service returned an invalid response.",
      ticker=ticker,
    )

  if result.get("status") == "error":
    error_type = str(result.get("error_type") or "").strip()
    if error_type == "invalid_input":
      raise CorpusReadinessGateError(
        422,
        "invalid_corpus_requirements",
        str(result.get("message") or result.get("error") or "Invalid corpus requirements."),
        details={"ticker": ticker},
      )
    log.error("Corpus readiness service error for %s: %s", ticker, result)
    raise _unavailable_error(
      "The corpus readiness service is temporarily unavailable.",
      ticker=ticker,
      upstream_code=error_type or "corpus_unavailable",
    )
  if result.get("status") != "success" or not isinstance(result.get("ready"), bool):
    raise _unavailable_error(
      "The corpus readiness service returned an invalid response.",
      ticker=ticker,
    )
  if result["ready"] is not True:
    raise CorpusReadinessGateError(
      409,
      "corpus_not_ready",
      f"Required corpus content is not ready for {ticker}.",
      details={"ticker": ticker, "readiness": dict(result)},
    )
  return dict(result)


def _normalize_requirements(
  payload: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
  normalized_payload = dict(payload)
  template = str(payload.get("pipeline_template") or "").strip().lower()
  raw_requirements = payload.get(CORPUS_REQUIREMENTS_FIELD)
  if raw_requirements is None:
    if template in CORPUS_GATED_PIPELINE_TEMPLATES:
      raise CorpusReadinessGateError(
        422,
        "missing_corpus_requirements",
        f"{template} batches require exact filing and transcript period declarations.",
        details={"pipeline_template": template},
      )
    return normalized_payload, ()
  if not isinstance(raw_requirements, list) or not raw_requirements:
    raise _invalid_requirements("corpus_requirements must be a non-empty array.")

  universe = _normalize_universe(payload.get("universe"))
  requirements: list[dict[str, Any]] = []
  seen_tickers: set[str] = set()
  for index, raw_requirement in enumerate(raw_requirements):
    if not isinstance(raw_requirement, dict):
      raise _invalid_requirements(
        f"corpus_requirements[{index}] must be an object."
      )
    unknown_keys = sorted(set(raw_requirement) - _REQUIREMENT_KEYS)
    if unknown_keys:
      raise _invalid_requirements(
        f"corpus_requirements[{index}] has unknown fields: {unknown_keys}."
      )
    try:
      ticker = canonicalize_ticker(raw_requirement.get("ticker"))
    except ValueError as exc:
      raise _invalid_requirements(
        f"corpus_requirements[{index}].ticker is invalid."
      ) from exc
    if ticker in seen_tickers:
      raise _invalid_requirements(
        f"corpus_requirements contains duplicate ticker {ticker}."
      )
    filings = _normalize_periods(
      raw_requirement.get("required_filings"),
      field=f"corpus_requirements[{index}].required_filings",
    )
    transcripts = _normalize_periods(
      raw_requirement.get("required_transcripts"),
      field=f"corpus_requirements[{index}].required_transcripts",
    )
    if not filings and not transcripts:
      raise _invalid_requirements(
        f"corpus_requirements[{index}] must declare at least one required period."
      )
    seen_tickers.add(ticker)
    requirements.append({
      "ticker": ticker,
      "required_filings": filings,
      "required_transcripts": transcripts,
    })

  if seen_tickers != set(universe) or len(requirements) != len(universe):
    raise _invalid_requirements(
      "corpus_requirements must contain exactly one entry for every universe ticker.",
      universe=list(universe),
      requirement_tickers=sorted(seen_tickers),
    )
  normalized_payload["universe"] = list(universe)
  normalized_payload[CORPUS_REQUIREMENTS_FIELD] = requirements
  return normalized_payload, tuple(requirements)


def _normalize_universe(value: Any) -> tuple[str, ...]:
  if not isinstance(value, list) or not value:
    raise _invalid_requirements(
      "A non-empty universe array is required when corpus_requirements are declared."
    )
  normalized: list[str] = []
  for index, ticker_raw in enumerate(value):
    try:
      ticker = canonicalize_ticker(ticker_raw)
    except ValueError as exc:
      raise _invalid_requirements(f"universe[{index}] is not a valid ticker.") from exc
    if ticker in normalized:
      raise _invalid_requirements(f"universe contains duplicate ticker {ticker}.")
    normalized.append(ticker)
  return tuple(normalized)


def _normalize_periods(value: Any, *, field: str) -> list[str]:
  if not isinstance(value, list):
    raise _invalid_requirements(f"{field} must be an array.")
  normalized: list[str] = []
  for index, raw_period in enumerate(value):
    if not isinstance(raw_period, str):
      raise _invalid_requirements(f"{field}[{index}] must be a string.")
    period = raw_period.strip().upper()
    if not _PERIOD_RE.fullmatch(period):
      raise _invalid_requirements(
        f"{field}[{index}] must match YYYY-Q1..Q4 or YYYY-FY."
      )
    if period in normalized:
      raise _invalid_requirements(f"{field} contains duplicate period {period}.")
    normalized.append(period)
  return normalized


def _invalid_requirements(message: str, **details: Any) -> CorpusReadinessGateError:
  return CorpusReadinessGateError(
    422,
    "invalid_corpus_requirements",
    message,
    details=details,
  )


def _unavailable_error(
  message: str,
  *,
  ticker: str | None = None,
  upstream_code: str | None = None,
) -> CorpusReadinessGateError:
  details = {
    key: value
    for key, value in {
      "ticker": ticker,
      "upstream_code": upstream_code,
    }.items()
    if value is not None
  }
  return CorpusReadinessGateError(
    503,
    "corpus_readiness_unavailable",
    message,
    details=details,
  )


__all__ = [
  "CORPUS_GATED_PIPELINE_TEMPLATES",
  "CORPUS_READINESS_TOOL",
  "CORPUS_REQUIREMENTS_FIELD",
  "CorpusReadinessGateError",
  "require_corpus_readiness",
]
