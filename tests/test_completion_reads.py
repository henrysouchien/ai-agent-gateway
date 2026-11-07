"""Raw completion contract: no terminal, count or receipt is a liveness verdict."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.completion_reads import (
  read_chat_completion,
  read_runner_completion,
  read_workflow_completion,
)
from agent_gateway.session import GatewaySession, SessionStream
from agent_gateway.task_registry import TaskState
from agent_gateway.runner_run_loop import _background_success_snapshot
from .test_runner_run_loop import _make_no_credential_runner, _NoCredentialProvider


def _runner() -> Any:
  runner = _make_no_credential_runner(_NoCredentialProvider())
  runner._runner_id = "runner-completion-test"
  return runner


def test_minimal_runner_keeps_missing_owners_unknown_and_preserves_journal() -> None:
  runner = _runner()
  runner._log.append({"type": "stream_complete", "text": "done"})
  snapshot = read_runner_completion(
    SimpleNamespace(), journal=runner._log,
    pending_approval={"count": 0, "refs": []},
  )

  assert snapshot.consistency == "unknown"
  assert snapshot.terminal is not None
  assert snapshot.terminal["event"]["text"] == "done"
  assert snapshot.children is None
  assert snapshot.durable is None
  assert snapshot.receipt is None
  assert snapshot.obligations == {"background": None, "session": None}
  assert set(snapshot.unknown) == {
    "background:AttributeError", "children:AttributeError",
    "session:AttributeError", "durable_owner:AttributeError",
    "durable:unavailable",
  }


@pytest.mark.parametrize("terminal", [
  {"type": "stream_complete", "text": "done"},
  {"type": "error", "error": "failed"},
  {"type": "stream_complete", "terminal_disposition": "interrupted"},
])
def test_terminal_does_not_erase_surviving_raw_children(terminal: dict[str, Any]) -> None:
  runner = _runner()
  states = [TaskState.RUNNING, TaskState.PARKED, TaskState.IDLE, TaskState.PENDING]
  for index, state in enumerate(states):
    entry = runner._task_registry.register("background_agent", task_id=f"child-{index}")
    entry.state = state
  runner._log.append(terminal)

  snapshot = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})

  assert snapshot.schema_version == 1
  assert snapshot.terminal is not None
  assert snapshot.journal is not None
  assert snapshot.children is not None
  assert snapshot.obligations is not None
  assert snapshot.terminal["event"] == terminal
  assert snapshot.journal["closed"] is True
  assert snapshot.children["count"] == 4
  assert [row["state"] for row in snapshot.children["children"]] == [state.value for state in states]
  native_blockers, _ = _background_success_snapshot(runner)
  assert snapshot.obligations["background"]["blockers"] == list(native_blockers)
  assert "quiescent" not in snapshot.model_dump()
  assert "settled" not in snapshot.model_dump()


def test_restart_keeps_observed_zero_distinct_from_known_zero() -> None:
  runner = _runner()
  before = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})
  assert before.children is not None
  assert before.children["count"] == 0
  runner._task_registry.load_from_events([])

  after = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})

  assert after.children is not None
  assert after.children["count"] is None
  assert after.children["observed_count"] == 0
  assert after.children["coverage"] == "unknown"
  assert "children:unknown_epoch" in after.unknown
  assert after.consistency == "unknown"


def test_pending_finalization_skill_result_and_delivery_are_native_facts() -> None:
  runner = _runner()
  entry = runner._task_registry.register(
    "background_agent", task_id="unsettled-child",
    required_skill_lifecycle={
      "schema_version": 2, "skill_run_id": "skill-1", "skill": "earnings-review",
      "scope": "ticker", "ticker": "PCTY", "portfolio_id": None,
    },
  )
  entry.state = TaskState.COMPLETED
  entry.completion_persistence_state = "committed"
  entry.completion_finalizer_detached = True
  entry.notification_delivery_state = "payload_omitted"
  runner._pending_background_result_acks["retrieve-1"] = (entry.task_id, entry.notification_generation)

  snapshot = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})

  assert snapshot.children is not None
  assert snapshot.obligations is not None
  child = snapshot.children["children"][0]
  assert child["completion_finalizer_detached"] is True
  assert child["required_skill_result_settled"] is False
  assert child["notification_delivery_state"] == "payload_omitted"
  background = snapshot.obligations["background"]
  assert background["pending_acknowledgements"]["count"] == 1
  assert set(background["blockers"]) >= {
    "background_required_skill_result_settlement", "background_finalization",
    "background_result_delivery", "background_result_acknowledgement",
  }
  entry.completion_persistence_state = "uncertain"
  uncertain = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})
  assert uncertain.children is not None
  assert uncertain.children["children"][0]["completion_persistence_state"] == "uncertain"


def test_durable_receipts_and_detach_are_not_live_log_close(tmp_path: Path) -> None:
  runner = _runner()
  durable = AgentSessionLog(path=tmp_path / "completion.jsonl")
  runner._agent_session_log = durable
  runner._log.append({"type": "stream_complete", "text": "done"})
  missing = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})
  assert missing.journal is not None
  assert missing.durable is not None
  assert missing.receipt is not None
  assert missing.journal["closed"] is True
  assert missing.durable["detach"] is None
  assert missing.receipt["refs"] == []
  durable.append_sync({"type": "stream_complete", "runner_id": runner._runner_id, "role": "writer"})
  durable.append_sync({"type": "detach", "runner_id": runner._runner_id, "role": "writer", "reason": "completed"})

  observed = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})

  assert observed.durable is not None
  assert observed.receipt is not None
  assert observed.durable == durable.completion_read(runner_id=runner._runner_id)
  assert observed.durable["detach"]["event"]["reason"] == "completed"
  assert observed.receipt["refs"] == observed.durable["receipts"]


def test_changed_source_is_reported_without_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = _runner()
  original = runner._log.completion_read
  reads = 0

  def changing_read() -> dict[str, Any]:
    nonlocal reads
    result = original()
    reads += 1
    if reads == 1:
      runner._task_registry.register("background_agent", task_id="arriving-child")
    return result

  monkeypatch.setattr(runner._log, "completion_read", changing_read)
  snapshot = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})

  assert snapshot.consistency == "changed"
  assert snapshot.children is not None
  assert snapshot.children["observed_count"] == 0
  assert reads == 2


def test_unknown_reader_is_not_empty_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  runner = _runner()
  runner._agent_session_log = AgentSessionLog(path=tmp_path / "broken.jsonl")

  def unavailable(**_kwargs: Any) -> Any:
    raise OSError("unavailable")

  monkeypatch.setattr(runner._agent_session_log, "completion_read", unavailable)
  snapshot = read_runner_completion(runner, pending_approval={"count": 0, "refs": []})

  assert snapshot.durable is None
  assert snapshot.receipt is None
  assert "durable:OSError" in snapshot.unknown


def test_chat_approval_park_and_missing_runner_are_explicit() -> None:
  runner = _runner()
  runner._log.append({"type": "stream_complete", "terminal_disposition": "awaiting_approval"})
  turn = SessionStream(event_log=runner._log, runner_task=None, runtime_owner=runner)
  session = SimpleNamespace(
    pending_tools={"tool-1": {"status": "approval_pending", "approval_id": "approval-1"}},
    active_turn=turn, last_completed_turn=None,
  )
  observed = read_chat_completion(session)
  assert observed.pending_approval == {
    "count": 1, "refs": [{"tool_call_id": "tool-1", "approval_id": "approval-1"}],
  }
  assert observed.terminal is not None
  assert observed.terminal["event"]["terminal_disposition"] == "awaiting_approval"
  turn.runtime_owner = None
  unknown = read_chat_completion(session)
  assert unknown.children is None
  assert unknown.obligations is None
  assert unknown.pending_approval is not None
  assert unknown.pending_approval["count"] == 1
  assert "children:unknown_epoch" in unknown.unknown


def test_workflow_manifest_and_ack_are_not_child_or_detach_completion() -> None:
  record = SimpleNamespace(
    event_channel_acknowledgement=SimpleNamespace(channel_id="channel", event_count=7, event_digest="digest"),
    terminal_manifest_committed=False, events_evidence_status="current",
  )
  receipt = {"run_id": "run-1", "disposition": "completed", "exit_code": 0}
  snapshot = read_workflow_completion(record, terminal_receipt=receipt, pending_approval_refs=[])
  assert snapshot.receipt is not None
  assert snapshot.durable is not None
  assert snapshot.terminal is not None
  assert snapshot.receipt["committed"] is False
  assert snapshot.durable["event_channel_acknowledgement"]["event_count"] == 7
  assert snapshot.durable["detach"] is None
  assert snapshot.children is None
  assert snapshot.obligations is None
  assert snapshot.consistency == "unknown"
  receipt["exit_code"] = 1
  assert snapshot.terminal["exit_code"] == 0


def test_control_chat_response_exposes_native_snapshot() -> None:
  from agent_gateway.control_plane.runs_helpers import _chat_run_from_session

  runner = _runner()
  entry = runner._task_registry.register("background_agent", task_id="idle-child")
  entry.state = TaskState.IDLE
  session = GatewaySession(
    session_id="chat-1", api_key_hash="test", created_at=1, expires_at=2,
    user_id="owner-1", owner_user_id="owner-1",
  )
  session.active_turn = SessionStream(
    event_log=runner._log, runner_task=None, runtime_owner=runner,
  )

  response = _chat_run_from_session(session)

  assert response.completion is not None
  assert response.completion.source == "chat"
  assert response.completion.children == runner._task_registry.completion_read()
  assert response.completion.journal == runner._log.completion_read()
  assert response.completion.obligations is not None
  native, _ = runner.background_completion_read()
  assert response.completion.obligations["background"] == native


def test_chat_pending_approval_change_marks_composition_changed(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = _runner()
  session = SimpleNamespace(
    pending_tools={}, active_turn=SessionStream(
      event_log=runner._log, runner_task=None, runtime_owner=runner,
    ), last_completed_turn=None,
  )
  original = runner.session_completion_read

  def changing_read() -> dict[str, Any]:
    session.pending_tools["tool-1"] = {
      "status": "approval_pending", "approval_id": "approval-1",
    }
    return original()

  monkeypatch.setattr(runner, "session_completion_read", changing_read)
  snapshot = read_chat_completion(session)
  assert snapshot.consistency == "changed"
  assert snapshot.pending_approval == {"count": 0, "refs": []}
