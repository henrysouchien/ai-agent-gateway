from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_gateway.autonomous_runner import AutonomousRegistry
from agent_gateway.control_plane.runs_helpers import AutonomousResumeRequest
from agent_gateway.control_plane.runs_resume_helpers import _build_autonomous_resume_context


def _paths(log_dir: Path, task_id: str) -> dict[str, Path]:
  return {
    "manifest": log_dir / f"{task_id}.task.json",
    "log": log_dir / f"{task_id}.log",
    "events": log_dir / f"{task_id}.events.jsonl",
    "operator": log_dir / f"{task_id}.operator-messages.jsonl",
    "approvals": log_dir / f"{task_id}.approval-decisions.jsonl",
  }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8",
  )


def _write_manifest(
  log_dir: Path,
  task_id: str,
  *,
  state: str,
  started_at: float = 100.0,
  completed_at: float | None = None,
  resumed_as: list[str] | None = None,
) -> dict[str, Any]:
  paths = _paths(log_dir, task_id)
  manifest = {
    "manifest_version": 1,
    "task_id": task_id,
    "control_run_id": f"run-{task_id}",
    "user_id": "user-h5",
    "user_email": "user@example.com",
    "profile": "analyst",
    "mode": "skill",
    "task": None,
    "skill": "earnings-review",
    "context": f"durable original context for {task_id}",
    "ticker": "AAPL",
    "channel": "tui",
    "dev_mode": False,
    "cmd": ["python3", "-m", "agent.autonomous"],
    "log_path": str(paths["log"]),
    "events_path": str(paths["events"]),
    "operator_inbox_path": str(paths["operator"]),
    "approval_decisions_path": str(paths["approvals"]),
    "started_at": started_at,
    "state": state,
    "exit_code": 1 if state == "interrupted" else 0,
    "error": "gateway restarted while run was active" if state == "interrupted" else None,
    "completed_at": completed_at,
    "resumed_from": None,
    "resumed_as": resumed_as or [],
  }
  paths["manifest"].write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
  return manifest


def _write_resume_evidence(log_dir: Path, task_id: str) -> None:
  paths = _paths(log_dir, task_id)
  paths["log"].write_text(f"{task_id} durable log tail line\n", encoding="utf-8")
  _write_jsonl(
    paths["events"],
    [
      {
        "type": "tool_call_start",
        "tool_call_id": f"{task_id}-tool",
        "tool_name": "fmp_fetch",
        "tool_input": {"ticker": "AAPL", "source": "durable event input"},
        "ts": 101,
      },
      {
        "type": "tool_call_complete",
        "tool_call_id": f"{task_id}-tool",
        "tool_name": "fmp_fetch",
        "result": {"revenue": f"{task_id} durable completed tool result"},
        "duration_ms": 12,
        "is_error": False,
        "ts": 102,
      },
      {"type": "text_delta", "text": f"{task_id} durable event tail", "ts": 103},
    ],
  )
  _write_jsonl(
    paths["operator"],
    [
      {
        "message_id": f"{task_id}-operator",
        "message": f"{task_id} durable operator message",
        "sent_at": 104,
      }
    ],
  )
  _write_jsonl(
    paths["approvals"],
    [
      {
        "approval_id": f"{task_id}-approval",
        "decision": "approved",
        "decided_at": 105,
      }
    ],
  )


def _all_resume_evidence_exists(log_dir: Path, task_id: str) -> bool:
  paths = _paths(log_dir, task_id)
  return all(path.exists() for path in paths.values())


def test_resume_context_rehydrates_from_manifest_and_evidence_files(tmp_path: Path) -> None:
  _write_manifest(tmp_path, "bg_10", state="interrupted", completed_at=200.0)
  _write_resume_evidence(tmp_path, "bg_10")

  registry = AutonomousRegistry(api_dir=tmp_path, log_dir=tmp_path)
  record = registry._tasks["bg_10"]

  context = _build_autonomous_resume_context(
    record,
    AutonomousResumeRequest(message="operator resume instruction from request"),
  )

  assert "state=interrupted" in context
  assert "operator resume instruction from request" in context
  assert "durable original context for bg_10" in context
  assert "bg_10 durable event tail" in context
  assert "bg_10 durable log tail line" in context
  assert "bg_10 durable operator message" in context
  assert "bg_10 durable completed tool result" in context


def test_registry_boot_keeps_resume_evidence_for_central_retention(monkeypatch, tmp_path: Path) -> None:
  from agent_gateway import autonomous_runner

  now = 1_000_000.0
  old = now - (8 * 86400)
  monkeypatch.setenv("AGENT_AUTONOMOUS_RUN_RETENTION_DAYS", "7")
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: now)

  _write_manifest(tmp_path, "bg_20", state="interrupted", started_at=old, completed_at=old)
  _write_resume_evidence(tmp_path, "bg_20")
  _write_manifest(tmp_path, "bg_21", state="completed", started_at=old, completed_at=old)
  _write_resume_evidence(tmp_path, "bg_21")

  registry = AutonomousRegistry(api_dir=tmp_path, log_dir=tmp_path)

  assert _all_resume_evidence_exists(tmp_path, "bg_20")
  assert "bg_20" in registry._tasks
  assert _all_resume_evidence_exists(tmp_path, "bg_21")
  assert "bg_21" in registry._tasks
