from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from agent_gateway.artifact_paths import canonicalize_ticker


CORPUS_READINESS_TOOL = "check_corpus_readiness"
CORPUS_REQUIREMENTS_FIELD = "corpus_requirements"

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
  """Observe declared corpus requirements without gating batch admission.

  Missing content and unavailable checks are warnings; only malformed external
  requirement declarations raise. An unavailable check is not a ready corpus.
  """

  normalized_payload, requirements = _normalize_requirements(payload)
  if not requirements:
    return normalized_payload, None

  gateway_config = getattr(app_state, "gateway_config", None)
  mcp_client = getattr(gateway_config, "mcp_client", None)
  unavailable_message = None
  if mcp_client is None or not callable(getattr(mcp_client, "call_tool", None)):
    unavailable_message = "The corpus readiness service is not configured."

  if unavailable_message is not None:
    checks = [
      _unavailable_check(requirement, unavailable_message)
      for requirement in requirements
    ]
  else:
    checks = await asyncio.gather(
      *(
        _check_one_requirement(mcp_client, requirement)
        for requirement in requirements
      )
    )
  warnings = []
  for requirement, check in zip(requirements, checks):
    if check["ready"] is True:
      continue
    unavailable = check["ready"] is None
    warnings.append({
      "code": "corpus_readiness_unavailable" if unavailable else "corpus_not_ready",
      "severity": "warning",
      "ticker": requirement["ticker"],
      "message": (
        check["message"] if unavailable
        else f"Required corpus content is not ready for {requirement['ticker']}."
      ),
    })
  status = "ready"
  if any(check["ready"] is None for check in checks):
    status = "unavailable"
  elif warnings:
    status = "not_ready"
  return normalized_payload, {
    "status": status,
    "checks": checks,
    "warnings": warnings,
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
    get_server_for_tool = getattr(mcp_client, "get_server_for_tool", None)
    if callable(get_server_for_tool) and get_server_for_tool(CORPUS_READINESS_TOOL) is None:
      return _unavailable_check(
        requirement,
        "The corpus readiness tool is not connected.",
      )
    result, error = await mcp_client.call_tool(CORPUS_READINESS_TOOL, tool_input)
  except asyncio.CancelledError:
    raise
  except Exception as exc:
    log.exception("Corpus readiness call failed for %s", ticker, exc_info=exc)
    return _unavailable_check(
      requirement,
      "The corpus readiness service could not complete the check.",
      upstream_error={"type": type(exc).__name__, "message": str(exc)},
    )

  if error is not None:
    log.error("Corpus readiness transport error for %s: %s", ticker, error)
    return _unavailable_check(
      requirement,
      "The corpus readiness service could not complete the check.",
      upstream_error=error,
    )
  if not isinstance(result, dict):
    return _unavailable_check(
      requirement,
      "The corpus readiness service returned an invalid response.",
      upstream_result=result,
    )

  if result.get("status") == "error":
    log.error("Corpus readiness service error for %s: %s", ticker, result)
    return _unavailable_check(
      requirement,
      "The corpus readiness service is temporarily unavailable.",
      upstream_result=result,
    )
  if result.get("status") != "success" or not isinstance(result.get("ready"), bool):
    return _unavailable_check(
      requirement,
      "The corpus readiness service returned an invalid response.",
      upstream_result=result,
    )
  return dict(result)


def _normalize_requirements(
  payload: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
  normalized_payload = dict(payload)
  raw_requirements = payload.get(CORPUS_REQUIREMENTS_FIELD)
  if raw_requirements is None:
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


def _unavailable_check(
  requirement: dict[str, Any],
  message: str,
  **details: Any,
) -> dict[str, Any]:
  return {
    **requirement,
    "status": "unavailable",
    "ready": None,
    "message": message,
    **details,
  }


__all__ = [
  "CORPUS_READINESS_TOOL",
  "CORPUS_REQUIREMENTS_FIELD",
  "CorpusReadinessGateError",
  "require_corpus_readiness",
]
