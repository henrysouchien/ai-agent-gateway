"""LH-26: staged-proposal handles surface on autonomous run responses."""

from __future__ import annotations

from typing import Any

from agent_gateway.control_plane.runs_helpers import _staged_proposals


def _staged_event(
  proposal_id: str,
  *,
  skill_run_id: str = "skill-run-1",
  status: str = "staged",
  **overrides: Any,
) -> dict[str, Any]:
  result: dict[str, Any] = {
    "proposal_id": proposal_id,
    "status": status,
    "expires_at": 200.0,
    "subcommand": "propose_monitoring_init",
    "ticker": "STWD",
    "readback": {"research_file_id": 2042},
  }
  result.update(overrides)
  return {
    "type": "skill_result_captured",
    "skill_run_id": skill_run_id,
    "fms_results": [result],
  }


def test_staged_proposal_fields_map_from_captured_result() -> None:
  proposals = _staged_proposals([_staged_event("prop-a")])
  assert len(proposals) == 1
  proposal = proposals[0]
  assert proposal.proposal_id == "prop-a"
  assert proposal.status == "staged"
  assert proposal.requires_apply is True
  assert proposal.expires_at == "1970-01-01T00:03:20Z"
  assert proposal.subcommand == "propose_monitoring_init"
  assert proposal.ticker == "STWD"
  assert proposal.research_file_id == 2042
  assert proposal.skill_run_id == "skill-run-1"


def test_duplicate_proposal_ids_dedupe_last_wins() -> None:
  events = [
    _staged_event("prop-a", expires_at=100.0),
    _staged_event("prop-a", expires_at=300.0),
  ]
  proposals = _staged_proposals(events)
  assert len(proposals) == 1
  assert proposals[0].expires_at == "1970-01-01T00:05:00Z"


def test_applied_proposals_are_not_listed() -> None:
  events = [
    _staged_event("prop-a"),
    _staged_event("prop-b"),
    _staged_event("prop-a", status="applied"),
  ]
  proposals = _staged_proposals(events)
  assert [proposal.proposal_id for proposal in proposals] == ["prop-b"]


def test_non_staged_statuses_are_ignored() -> None:
  events = [
    _staged_event("prop-a", status="noop"),
    _staged_event("prop-b", status="error"),
  ]
  assert _staged_proposals(events) == []


def test_pathological_expires_at_values_do_not_crash() -> None:
  events = [
    _staged_event("prop-nan", expires_at=float("nan")),
    _staged_event("prop-inf", expires_at=float("inf")),
    _staged_event("prop-neg-inf", expires_at=float("-inf")),
    _staged_event("prop-overflow", expires_at=1e18),
    _staged_event("prop-int-overflow", expires_at=10**400),
    _staged_event("prop-bool", expires_at=True),
  ]
  proposals = _staged_proposals(events)
  assert [proposal.proposal_id for proposal in proposals] == [
    "prop-nan",
    "prop-inf",
    "prop-neg-inf",
    "prop-overflow",
    "prop-int-overflow",
    "prop-bool",
  ]
  assert all(proposal.expires_at is None for proposal in proposals)


def test_malformed_entries_are_tolerated() -> None:
  events = [
    {"type": "other"},
    {"type": "skill_result_captured", "skill_run_id": "s", "fms_results": "nope"},
    {
      "type": "skill_result_captured",
      "fms_results": [
        None,
        {"status": "staged"},
        {"proposal_id": "", "status": "staged"},
        {
          "proposal_id": "prop-a",
          "status": "staged",
          "expires_at": "soon",
          "readback": None,
          "subcommand": None,
          "ticker": None,
        },
      ],
    },
  ]
  proposals = _staged_proposals(events)
  assert len(proposals) == 1
  proposal = proposals[0]
  assert proposal.proposal_id == "prop-a"
  assert proposal.expires_at is None
  assert proposal.subcommand is None
  assert proposal.ticker is None
  assert proposal.research_file_id is None
  assert proposal.skill_run_id is None
