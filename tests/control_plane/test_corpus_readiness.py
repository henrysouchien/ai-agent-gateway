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
    *responses: tuple[Any | None, dict[str, Any] | None],
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
    return self.responses.pop(0)


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
  assert readiness == {"status": "ready", "checks": [_ready_result()]}
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


def test_named_gated_workflow_requires_exact_declaration() -> None:
  payload = {
    "source": "explicit_ticker",
    "universe": ["PCTY"],
    "pipeline_template": "valuation-ready",
  }

  with pytest.raises(CorpusReadinessGateError) as caught:
    asyncio.run(require_corpus_readiness(payload, app_state=SimpleNamespace()))

  assert caught.value.status_code == 422
  assert caught.value.code == "missing_corpus_requirements"


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


def test_not_ready_returns_conflict_with_selector_evidence() -> None:
  not_ready = {
    **_ready_result(),
    "ready": False,
    "found_transcripts": ["2026-Q1"],
    "missing_transcripts": ["2026-Q2"],
  }
  mcp_client = _FakeMcpClient((not_ready, None))

  with pytest.raises(CorpusReadinessGateError) as caught:
    asyncio.run(require_corpus_readiness(_payload(), app_state=_state(mcp_client)))

  assert caught.value.status_code == 409
  assert caught.value.code == "corpus_not_ready"
  assert caught.value.details["readiness"]["missing_transcripts"] == ["2026-Q2"]


@pytest.mark.parametrize(
  "mcp_client",
  [
    _FakeMcpClient((_ready_result(), None), connected=False),
    _FakeMcpClient((None, {"code": "tool_error", "message": "offline"})),
    _FakeMcpClient(({"status": "error", "error_type": "corpus_unavailable"}, None)),
    _FakeMcpClient(({"status": "success", "ready": "yes"}, None)),
  ],
)
def test_unavailable_or_invalid_selector_response_fails_closed(
  mcp_client: _FakeMcpClient,
) -> None:
  with pytest.raises(CorpusReadinessGateError) as caught:
    asyncio.run(require_corpus_readiness(_payload(), app_state=_state(mcp_client)))

  assert caught.value.status_code == 503
  assert caught.value.code == "corpus_readiness_unavailable"
