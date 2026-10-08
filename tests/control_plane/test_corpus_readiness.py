from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway.control_plane.corpus_readiness import (
  CORPUS_READINESS_TOOL,
  CorpusReadinessGateError,
  require_corpus_readiness,
)


class _FakeMcpClient:
  def __init__(
    self,
    *responses: tuple[Any | None, dict[str, Any] | None] | BaseException,
    connected: bool = True,
  ) -> None:
    self.responses = list(responses)
    self.connected = connected
    self.calls: list[tuple[str, dict[str, Any]]] = []

  def get_server_for_tool(self, name: str) -> str | None:
    if self.connected and name == CORPUS_READINESS_TOOL:
      return "research-corpus-mcp"
    return None

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any],
  ) -> tuple[Any | None, dict[str, Any] | None]:
    self.calls.append((name, dict(tool_input)))
    response = self.responses.pop(0)
    if isinstance(response, BaseException):
      raise response
    return response


def _state(mcp_client: Any) -> SimpleNamespace:
  return SimpleNamespace(gateway_config=SimpleNamespace(mcp_client=mcp_client))


def _payload() -> dict[str, Any]:
  return {
    "source": "explicit_ticker",
    "universe": ["pcty"],
    "pipeline_template": "valuation-ready",
    "budget_usd": 25.0,
    "corpus_requirements": [{
      "ticker": "pcty",
      "required_filings": ["2025-fy", "2026-q2"],
      "required_transcripts": ["2026-q1", "2026-q2"],
    }],
  }


def _ready_result() -> dict[str, Any]:
  return {
    "status": "success",
    "ticker": "PCTY",
    "ready": True,
    "required_filings": ["2025-FY", "2026-Q2"],
    "required_transcripts": ["2026-Q1", "2026-Q2"],
    "found_filings": ["2025-FY", "2026-Q2"],
    "found_transcripts": ["2026-Q1", "2026-Q2"],
    "missing_filings": [],
    "missing_transcripts": [],
    "unavailable_filings": [],
    "unavailable_transcripts": [],
  }


def test_ready_declaration_is_canonicalized_and_checked_before_dispatch() -> None:
  mcp_client = _FakeMcpClient((_ready_result(), None))

  normalized, readiness = asyncio.run(
    require_corpus_readiness(_payload(), app_state=_state(mcp_client))
  )

  assert normalized["universe"] == ["PCTY"]
  assert normalized["corpus_requirements"] == [{
    "ticker": "PCTY",
    "required_filings": ["2025-FY", "2026-Q2"],
    "required_transcripts": ["2026-Q1", "2026-Q2"],
  }]
  assert readiness == {"status": "ready", "checks": [_ready_result()], "warnings": []}
  assert mcp_client.calls == [(
    CORPUS_READINESS_TOOL,
    {
      "ticker": "PCTY",
      "required_filings": ["2025-FY", "2026-Q2"],
      "required_transcripts": ["2026-Q1", "2026-Q2"],
    },
  )]


def test_ungated_batch_without_declaration_does_not_call_mcp() -> None:
  payload = {"source": "quality_screen", "universe": ["MSFT"]}

  normalized, readiness = asyncio.run(
    require_corpus_readiness(payload, app_state=SimpleNamespace())
  )

  assert normalized == payload
  assert readiness is None


def test_valuation_ready_without_declaration_proceeds_without_corpus_check() -> None:
  payload = {
    "source": "explicit_ticker",
    "universe": ["PCTY"],
    "pipeline_template": "valuation-ready",
  }

  normalized, readiness = asyncio.run(
    require_corpus_readiness(payload, app_state=SimpleNamespace())
  )

  assert normalized == payload
  assert readiness is None


@pytest.mark.parametrize(
  "mutation, message",
  [
    (
      lambda payload: payload["corpus_requirements"][0].update(
        {"required_filings": "2025-FY"}
      ),
      "required_filings must be an array",
    ),
    (
      lambda payload: payload.update({"universe": ["PCTY", "MSFT"]}),
      "exactly one entry for every universe ticker",
    ),
    (
      lambda payload: payload["corpus_requirements"][0].update(
        {"required_transcripts": ["latest"]}
      ),
      "must match YYYY-Q1..Q4 or YYYY-FY",
    ),
  ],
)
def test_invalid_declarations_fail_before_mcp(
  mutation: Any,
  message: str,
) -> None:
  payload = _payload()
  mutation(payload)
  mcp_client = _FakeMcpClient((_ready_result(), None))

  with pytest.raises(CorpusReadinessGateError, match=message) as caught:
    asyncio.run(require_corpus_readiness(payload, app_state=_state(mcp_client)))

  assert caught.value.status_code == 422
  assert caught.value.code == "invalid_corpus_requirements"
  assert mcp_client.calls == []


def test_not_ready_is_a_warning_and_dispatch_proceeds() -> None:
  not_ready = {
    **_ready_result(),
    "ready": False,
    "found_transcripts": ["2026-Q1"],
    "missing_transcripts": ["2026-Q2"],
  }
  mcp_client = _FakeMcpClient((not_ready, None))

  _normalized, readiness = asyncio.run(
    require_corpus_readiness(_payload(), app_state=_state(mcp_client))
  )

  assert readiness is not None
  assert readiness["status"] == "not_ready"
  assert readiness["checks"] == [not_ready]
  assert readiness["warnings"] == [{
    "code": "corpus_not_ready",
    "severity": "warning",
    "ticker": "PCTY",
    "message": "Required corpus content is not ready for PCTY.",
  }]


@pytest.mark.parametrize(
  "mcp_client, message",
  [
    (None, "not configured"),
    (SimpleNamespace(), "not configured"),
    (_FakeMcpClient(connected=False), "not connected"),
  ],
)
def test_missing_checker_reports_each_ticker_as_unknown(
  mcp_client: Any,
  message: str,
) -> None:
  payload = _payload()
  payload["universe"].append("MSFT")
  payload["corpus_requirements"].append({
    "ticker": "MSFT",
    "required_filings": ["2025-FY"],
    "required_transcripts": [],
  })

  normalized, readiness = asyncio.run(
    require_corpus_readiness(payload, app_state=_state(mcp_client))
  )

  assert normalized["universe"] == ["PCTY", "MSFT"]
  assert readiness is not None
  assert readiness["status"] == "unavailable"
  assert len(readiness["checks"]) == len(readiness["warnings"]) == 2
  for requirement, check, warning in zip(
    normalized["corpus_requirements"], readiness["checks"], readiness["warnings"],
  ):
    assert check["ticker"] == warning["ticker"] == requirement["ticker"]
    assert check["required_filings"] == requirement["required_filings"]
    assert check["required_transcripts"] == requirement["required_transcripts"]
    assert check["ready"] is None
    assert check["status"] == "unavailable"
    assert message in check["message"]
    assert warning["code"] == "corpus_readiness_unavailable"
    assert warning["severity"] == "warning"
    assert warning["message"] == check["message"]
  if isinstance(mcp_client, _FakeMcpClient):
    assert mcp_client.calls == []


@pytest.mark.parametrize(
  "response, details",
  [
    (
      (None, {"code": "tool_error", "message": "offline", "data": {"retry": "reconnect"}}),
      {"upstream_error": {
        "code": "tool_error", "message": "offline", "data": {"retry": "reconnect"},
      }},
    ),
    (
      ({"status": "error", "error_type": "corpus_unavailable", "message": "index offline"}, None),
      {"upstream_result": {
        "status": "error", "error_type": "corpus_unavailable", "message": "index offline",
      }},
    ),
    (
      ({"status": "error", "error_type": "invalid_input", "message": "checker rejected period"}, None),
      {"upstream_result": {
        "status": "error", "error_type": "invalid_input", "message": "checker rejected period",
      }},
    ),
    (
      ({"status": "success", "ready": "yes"}, None),
      {"upstream_result": {"status": "success", "ready": "yes"}},
    ),
    ((["invalid response"], None), {"upstream_result": ["invalid response"]}),
    ((None, None), {"upstream_result": None}),
    (
      RuntimeError("connection reset"),
      {"upstream_error": {"type": "RuntimeError", "message": "connection reset"}},
    ),
  ],
)
def test_checker_failure_is_unknown_and_preserves_upstream_information(
  response: Any,
  details: dict[str, Any],
) -> None:
  normalized, readiness = asyncio.run(
    require_corpus_readiness(_payload(), app_state=_state(_FakeMcpClient(response)))
  )

  assert normalized["universe"] == ["PCTY"]
  assert readiness is not None
  assert readiness["status"] == "unavailable"
  check = readiness["checks"][0]
  assert check["ticker"] == "PCTY"
  assert check["ready"] is None
  assert check["status"] == "unavailable"
  for key, value in details.items():
    assert check[key] == value
  assert readiness["warnings"] == [{
    "code": "corpus_readiness_unavailable",
    "severity": "warning",
    "ticker": "PCTY",
    "message": check["message"],
  }]


def test_mixed_results_preserve_ready_not_ready_and_unknown_observations() -> None:
  payload = _payload()
  payload["universe"].extend(["MSFT", "AAPL"])
  for ticker in ("MSFT", "AAPL"):
    payload["corpus_requirements"].append({
      "ticker": ticker,
      "required_filings": ["2025-FY"],
      "required_transcripts": [],
    })
  not_ready = {
    "status": "success",
    "ticker": "AAPL",
    "ready": False,
    "missing_filings": ["2025-FY"],
  }
  error = {"code": "tool_error", "message": "MSFT index unavailable"}
  mcp_client = _FakeMcpClient((_ready_result(), None), (None, error), (not_ready, None))

  normalized, readiness = asyncio.run(
    require_corpus_readiness(payload, app_state=_state(mcp_client))
  )

  assert normalized["universe"] == ["PCTY", "MSFT", "AAPL"]
  assert readiness is not None
  assert readiness["status"] == "unavailable"
  assert readiness["checks"][0] == _ready_result()
  assert readiness["checks"][1]["ticker"] == "MSFT"
  assert readiness["checks"][1]["ready"] is None
  assert readiness["checks"][1]["upstream_error"] == error
  assert readiness["checks"][2] == not_ready
  assert [(warning["ticker"], warning["code"]) for warning in readiness["warnings"]] == [
    ("MSFT", "corpus_readiness_unavailable"),
    ("AAPL", "corpus_not_ready"),
  ]


def test_cancellation_propagates_instead_of_becoming_a_warning() -> None:
  mcp_client = _FakeMcpClient(asyncio.CancelledError())

  with pytest.raises(asyncio.CancelledError):
    asyncio.run(require_corpus_readiness(_payload(), app_state=_state(mcp_client)))
