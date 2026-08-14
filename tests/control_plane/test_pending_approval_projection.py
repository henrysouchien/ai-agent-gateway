from __future__ import annotations

from types import SimpleNamespace

from agent_gateway.control_plane.runs_helpers import _latest_tool_result, _pending_approval


def test_pending_approval_projects_trusted_change_without_agent_judgment() -> None:
  planned_change = {
    "schema_version": "planned-change-review.v1",
    "change_set_id": "change-set-1",
    "change_hash": "a" * 64,
    "intent": {"subcommand": "persist_business_model"},
    "target": {"ticker": "MSFT", "research_file_id": 1},
  }
  session = SimpleNamespace(
    pending_tools={
      "call-1": {
        "status": "approval_pending",
        "approval_id": "approval-1",
        "tool_name": "fms_persist_business_model",
        "planned_change": planned_change,
        "resolved_qualifier": "qual-1",
        "reason": "review exact plan",
        "allow_persistent_approval": False,
        "requested_at": 1_786_233_600,
      }
    }
  )

  response = _pending_approval(session)

  assert response is not None
  assert response.tool_input == {}
  assert response.planned_change == planned_change


def test_latest_tool_result_is_bounded_and_preserves_terminal_outcome() -> None:
  summary = _latest_tool_result(
    [
      {
        "type": "tool_call_complete",
        "tool_call_id": "tool-1",
        "tool_name": "fms_persist_business_model",
        "is_error": False,
        "result": {
          "status": "staged",
          "subcommand": "persist_business_model",
          "artifact_ref": "artifacts/PCTY/business-model.json",
          "proposal_id": "proposal-1",
          "verdict_echo": {"verdict": "BM_CONSTRUCTED"},
          "readback": {
            "typed_outputs": {
              "business_model_stage_receipt": {"status": "accepted"},
              "large": "x" * 100_000,
            }
          },
        },
      }
    ]
  )

  assert summary is not None
  assert summary.succeeded is True
  assert summary.status == "staged"
  assert summary.verdict == "BM_CONSTRUCTED"
  assert summary.stage_receipt_status == "accepted"
  assert "readback" not in summary.model_dump()


def test_latest_tool_result_extracts_verdict_from_business_model_envelope() -> None:
  summary = _latest_tool_result(
    [
      {
        "type": "tool_call_complete",
        "tool_call_id": "tool-1",
        "tool_name": "fms_persist_business_model",
        "is_error": False,
        "result": {
          "status": "staged",
          "verdict": {
            "skill": "business-model-construction",
            "verdict": "BM_CONSTRUCTED",
            "validation": {"large": "x" * 100_000},
          },
          "verdict_echo": {"verdict": "BM_CONSTRUCTED"},
          "readback": {
            "typed_outputs": {
              "business_model_stage_receipt": {"status": "accepted"},
            }
          },
        },
      }
    ]
  )

  assert summary is not None
  assert summary.verdict == "BM_CONSTRUCTED"
  assert summary.stage_receipt_status == "accepted"


def _forecast_terminal_result(*, readback_status: str | None, receipt_status: str | None):
  readback = {}
  if readback_status is not None:
    readback["stage_receipt_status"] = readback_status
  receipt = {}
  if receipt_status is not None:
    receipt["stage_receipt"] = {"status": receipt_status, "large": "x" * 100_000}
  return _latest_tool_result(
    [
      {
        "type": "tool_call_complete",
        "tool_call_id": "tool-forecast-1",
        "tool_name": "fms_persist_forecast_assumption_set",
        "is_error": False,
        "result": {
          "status": "staged",
          "gate_code": "INSUFFICIENT_DATA",
          "verdict_echo": {"verdict": "ASSUMPTIONS_INCOMPLETE"},
          "readback": readback,
          "receipt": receipt,
        },
      }
    ]
  )


def test_latest_tool_result_extracts_forecast_stage_receipt_from_readback() -> None:
  summary = _forecast_terminal_result(
    readback_status="incomplete",
    receipt_status=None,
  )

  assert summary is not None
  assert summary.verdict == "ASSUMPTIONS_INCOMPLETE"
  assert summary.stage_receipt_status == "incomplete"


def test_latest_tool_result_extracts_forecast_stage_receipt_from_persisted_receipt() -> None:
  summary = _forecast_terminal_result(
    readback_status=None,
    receipt_status="blocked",
  )

  assert summary is not None
  assert summary.stage_receipt_status == "blocked"


def test_latest_tool_result_fails_closed_on_conflicting_stage_receipts() -> None:
  summary = _forecast_terminal_result(
    readback_status="accepted",
    receipt_status="incomplete",
  )

  assert summary is not None
  assert summary.stage_receipt_status is None
