from __future__ import annotations

from pathlib import Path

from agent_gateway.autonomous_launch_envelope import AutonomousControlAuthority
from agent_gateway.autonomous_runner import AutonomousTask
from agent_gateway.capability_binding import CapabilityBind
from agent_gateway.skill_limits import SkillExecutionLimits
from agent_gateway.control_plane.runs_helpers import (
  _autonomous_result_refs,
  _autonomous_run_from_task,
  _autonomous_terminal_receipt,
)

from .manifest_helpers import write_v6_manifest


def _run_record(
  tmp_path: Path,
  *,
  state: str,
  completed_at: float | None,
  exit_code: int | None,
  error: str | None,
  terminal_reason: str | None = None,
  control_run_id: str = "bg_1",
  events: list[dict] | None = None,
) -> AutonomousTask:
  manifest = write_v6_manifest(
    tmp_path / "autonomous",
    "bg_1",
  )
  return AutonomousTask(
    task_id="bg_1",
    control_run_id=control_run_id,
    session_id="bg_1",
    channel_id=manifest["channel_id"],
    user_id="user-1",
    user_email="user@example.com",
    role="owner",
    profile="analyst",
    mode="skill",
    task=None,
    skill="thesis-review",
    pack=None,
    deliver=True,
    context=None,
    ticker="FOO",
    channel="cli",
    dev_mode=False,
    dispatch_scope=None,
    cmd=list(manifest["cmd"]),
    log_path=Path(manifest["log_path"]),
    operator_inbox_path=Path(manifest["operator_inbox_path"]),
    approval_decisions_path=None,
    control_authority=AutonomousControlAuthority.from_receipt(
      manifest["control_authority"]
    ),
    owner_lease_path=Path(manifest["owner_lease_path"]),
    owner_lease_device=manifest["owner_lease_device"],
    owner_lease_inode=manifest["owner_lease_inode"],
    started_at=1784980000,
    skill_resume_allowed=False,
    admitted_skill_execution_limits=SkillExecutionLimits(None, None, None),
    state=state,
    completed_at=completed_at,
    exit_code=exit_code,
    error=error,
    terminal_reason=terminal_reason,
    event_lines=list(events or []),
    owner_user_id="user-1",
    raw_user_id="raw-user-1",
    user_slug="user-1",
    risk_user_id=1,
    user_aliases=["user-1"],
    identity_status="resolved",
    capability_bind=CapabilityBind.model_validate(
      manifest["capability_bind"]
    ),
  )


def test_terminal_receipt_is_exact_and_carries_stable_result_references(
  tmp_path: Path,
) -> None:
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
  record = _run_record(
    tmp_path,
    state="completed",
    completed_at=1784980800,
    exit_code=0,
    error=None,
    control_run_id="bg/receipt 1",
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


def test_terminal_receipt_is_absent_until_record_is_settled(
  tmp_path: Path,
) -> None:
  record = _run_record(
    tmp_path,
    state="running",
    completed_at=None,
    exit_code=None,
    error=None,
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


def test_terminal_receipt_carries_typed_writer_lease_reason(
  tmp_path: Path,
) -> None:
  record = _run_record(
    tmp_path,
    state="completed",
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


def test_run_projection_exposes_exact_failed_receipt_and_nonterminal_none(
  tmp_path: Path,
) -> None:
  failed = _autonomous_run_from_task(
    _run_record(
      tmp_path,
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
      tmp_path,
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
