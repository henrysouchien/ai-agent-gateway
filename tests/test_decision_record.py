from types import SimpleNamespace

from agent_gateway.decision_record import build_decision_record, run_pointer_from_context


def test_decision_record_preserves_seam_vocabulary_and_join_keys():
  object_ref = {"proposal_id": "proposal-1", "op_hash": "hash-1"}
  run = {"tool_call_id": "call-1", "request_id": "request-1"}
  decision = build_decision_record(
    actor="agent",
    verdict="dismissed",
    object_kind="patch_proposal",
    object_ref=object_ref,
    reason="  Not the intended update.  ",
    run=run,
    decided_at=123.5,
  )
  assert decision == {
    "schema_version": 1,
    "actor": "agent",
    "verdict": "dismissed",
    "object_kind": "patch_proposal",
    "object_ref": object_ref,
    "reason": "Not the intended update.",
    "run": run,
    "decided_at": 123.5,
  }
  object_ref["op_hash"] = "later"
  run["tool_call_id"] = "later"
  assert decision["object_ref"]["op_hash"] == "hash-1"
  assert decision["run"]["tool_call_id"] == "call-1"


def test_decision_defaults_timestamp_and_does_not_reject_unknown_vocabulary(monkeypatch):
  monkeypatch.setattr("agent_gateway.decision_record.time.time", lambda: 456.0)
  decision = build_decision_record(
    actor="new-actor", verdict="new-verdict", object_kind="new-output", object_ref={}, reason="  "
  )
  assert decision["decided_at"] == 456.0
  assert decision["actor"] == "new-actor"
  assert decision["verdict"] == "new-verdict"
  assert decision["reason"] is None
  assert "run" not in decision
  assert build_decision_record(
    actor="agent", verdict="kept", object_kind="output", object_ref={}, decided_at="2026-10-16"
  )["decided_at"] == "2026-10-16"


def test_run_pointer_extracts_only_present_join_keys_from_objects_and_mappings():
  values = {
    "session_id": "session-1",
    "request_id": "request-1",
    "skill_run_id": "skill-1",
    "control_run_id": "control-1",
    "tool_call_id": "call-1",
  }
  assert run_pointer_from_context(SimpleNamespace(**values, user_id="alice")) == values
  assert run_pointer_from_context({**values, "user_id": "alice"}) == values
  assert run_pointer_from_context(SimpleNamespace(request_id="", skill_run_id=None)) is None
  assert run_pointer_from_context(None) is None
