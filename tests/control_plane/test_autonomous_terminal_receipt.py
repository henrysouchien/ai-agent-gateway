from __future__ import annotations

from types import SimpleNamespace

from agent_gateway.control_plane.runs_helpers import (
  _autonomous_result_refs,
  _autonomous_run_from_task,
  _autonomous_terminal_receipt,
)


def _run_record(
  *,
  state: str,
  completed_at: float | None,
  exit_code: int | None,
  error: str | None,
  events: list[dict] | None = None,
) -> SimpleNamespace:
  return SimpleNamespace(
    task_id="bg_1",
    control_run_id="bg_1",
    user_id="user-1",
    owner_user_id="user-1",
    raw_user_id="raw-user-1",
    user_slug="user-1",
    risk_user_id=1,
    user_email="user@example.com",
    user_aliases=["user-1"],
    identity_status="resolved",
    profile="analyst",
    mode="skill",
    skill="thesis-review",
    task=None,
    ticker="FOO",
    channel="cli",
    state=state,
    started_at=1784980000,
    completed_at=completed_at,
    exit_code=exit_code,
    error=error,
    terminal_reason=None,
    max_budget_usd=10.0,
    event_lines=list(events or []),
    operator_inbox_path=None,
    proc=None,
    capability_bind=None,
    execution_transport=None,
    resumed_from=None,
    resumed_as=[],
    dispatch_scope=None,
    schedule_id=None,
    schedule_name=None,
  )


def test_terminal_receipt_is_exact_and_carries_stable_result_references() -> None:
  events = [
    {
      "type": "skill_result_captured",
      "skill_run_id": "skill/run 1",
      "artifact_refs": ["artifact://one", "artifact://one", "", 7],
      "proposal_ids": ["proposal-1", "proposal-1"],
      "output_memory_file": "skills/review/output.md",
    },
    {
      "type": "skill_result_captured",
      "skill_run_id": "skill/run 1",
      "artifact_refs": ["artifact://two"],
      "output_memory_file": "skills/review/output.md",
    },
  ]
  record = SimpleNamespace(
    control_run_id="bg/receipt 1",
    completed_at=1784980800,
    exit_code=0,
    error=None,
    terminal_reason=None,
  )

  receipt = _autonomous_terminal_receipt(
    record,
    state="completed",
    events=events,
  )

  assert receipt is not None
  assert receipt.model_dump() == {
    "run_id": "bg/receipt 1",
    "disposition": "completed",
    "exit_code": 0,
    "error": None,
    "terminal_reason": None,
    "completed_at": "2026-07-25T12:00:00Z",
    "log_ref": "/control/runs/bg%2Freceipt%201/logs",
    "result_refs": [
      {
        "kind": "skill_run",
        "ref": "skill/run 1",
        "skill_run_id": "skill/run 1",
      },
      {
        "kind": "artifact",
        "ref": "artifact://one",
        "skill_run_id": "skill/run 1",
      },
      {
        "kind": "proposal",
        "ref": "proposal-1",
        "skill_run_id": "skill/run 1",
      },
      {
        "kind": "output_memory",
        "ref": "skills/review/output.md",
        "skill_run_id": "skill/run 1",
      },
      {
        "kind": "artifact",
        "ref": "artifact://two",
        "skill_run_id": "skill/run 1",
      },
    ],
  }


def test_terminal_receipt_is_absent_until_record_is_settled() -> None:
  record = SimpleNamespace(
    control_run_id="bg_1",
    completed_at=None,
    exit_code=None,
    error=None,
    terminal_reason=None,
  )

  assert _autonomous_terminal_receipt(
    record,
    state="running",
    events=[],
  ) is None
  assert _autonomous_terminal_receipt(
    record,
    state="completed",
    events=[],
  ) is None


def test_terminal_receipt_carries_typed_writer_lease_reason() -> None:
  record = SimpleNamespace(
    control_run_id="bg_1",
    completed_at=1784980800,
    exit_code=0,
    error=None,
    terminal_reason="writer_lease_already_held",
  )

  receipt = _autonomous_terminal_receipt(
    record,
    state="completed",
    events=[],
  )

  assert receipt is not None
  assert receipt.terminal_reason == "writer_lease_already_held"


def test_run_projection_exposes_exact_failed_receipt_and_nonterminal_none() -> None:
  failed = _autonomous_run_from_task(
    _run_record(
      state="failed",
      completed_at=1784980800,
      exit_code=75,
      error="provider unavailable",
      events=[
        {
          "type": "skill_result_captured",
          "skill_run_id": "skill-run-failed",
          "artifact_refs": ["artifact://partial"],
          "output_memory_file": None,
        }
      ],
    )
  )

  assert failed.exit_code == 75
  assert failed.error == "provider unavailable"
  assert failed.terminal_receipt is not None
  assert failed.terminal_receipt.model_dump() == {
    "run_id": "bg_1",
    "disposition": "failed",
    "exit_code": 75,
    "error": "provider unavailable",
    "terminal_reason": None,
    "completed_at": "2026-07-25T12:00:00Z",
    "log_ref": "/control/runs/bg_1/logs",
    "result_refs": [
      {
        "kind": "skill_run",
        "ref": "skill-run-failed",
        "skill_run_id": "skill-run-failed",
      },
      {
        "kind": "artifact",
        "ref": "artifact://partial",
        "skill_run_id": "skill-run-failed",
      },
    ],
  }

  running = _autonomous_run_from_task(
    _run_record(
      state="running",
      completed_at=None,
      exit_code=None,
      error=None,
    )
  )
  assert running.terminal_receipt is None
  assert running.exit_code is None
  assert running.error is None


def test_result_references_ignore_uncaptured_and_malformed_values() -> None:
  assert _autonomous_result_refs(
    [
      {"type": "skill_run_started", "skill_run_id": "not-a-result"},
      {
        "type": "skill_result_captured",
        "skill_run_id": "",
        "artifact_refs": "not-a-list",
        "output_memory_file": None,
      },
    ]
  ) == []
