from __future__ import annotations

import hashlib

from agent_workflow_contracts import (
  ActivityHandle,
  AgentCompletionEnvelope,
  AgentOperationRef,
  AttemptRef,
  CanonicalProjection,
  ContractRef,
  ContentHandle,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  ProjectionInline,
  SettlementProjection,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultRef,
  TaskResultValues,
  TranscriptHandle,
  canonical_json_bytes,
)

from agent_gateway.event_adapter import adapt_event
from agent_gateway.runner_session_events import build_agent_completion_event
from agent_gateway.task_registry import TaskRegistry, TaskState


_DIGEST = f"sha256:{'a' * 64}"


def _contract() -> ContractRef:
  return ContractRef(
    namespace="test",
    name="projection",
    version="1",
    digest=_DIGEST,
  )


def _result(task_id: str = "bg_7") -> TaskResult:
  value = {"status": "complete"}
  raw = canonical_json_bytes(value)
  content_sha = hashlib.sha256(raw).hexdigest()
  handle = ContentHandle(
    content_id=f"sha256:{content_sha}",
    content_sha256=content_sha,
    content_bytes=len(raw),
    content_chars=len(raw.decode("utf-8")),
    contract=_contract(),
    media_type="application/json",
    encoding="utf-8",
    retention="durable",
  )
  operation = AgentOperationRef(
    namespace="operation",
    name="explore",
    version="1",
    digest=_DIGEST,
  )
  logical = OrdinaryDelegationTaskRef(
    delegation_id="delegation-1",
    operation=operation,
  )
  attempt = AttemptRef(
    attempt_number=1,
    attempt_id="attempt-1",
    physical_task_id=task_id,
  )
  return TaskResult(
    task_result_id="task-result-1",
    logical_task=logical,
    attempt=attempt,
    execution=ExecutionSettlement(status="succeeded"),
    values=TaskResultValues(
      projection=CanonicalProjection(
        contract=_contract(),
        content=handle,
        inline_view=value,
      )
    ),
    observation=TaskObservation(
      transcript=TranscriptHandle(kind="child_transcript", owner_id=task_id),
      activity=ActivityHandle(kind="child_activity", owner_id=task_id),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=_DIGEST,
      model_bind_digest=_DIGEST,
      capability_binding_digest=_DIGEST,
      tool_grant_digest=_DIGEST,
    ),
  )


def _envelope(result: TaskResult, *, message_id: str = "message-1") -> AgentCompletionEnvelope:
  assert result.values.projection is not None
  projection = result.values.projection
  return AgentCompletionEnvelope(
    message_id=message_id,
    task_result_ref=TaskResultRef.from_result(result),
    settlement_projection=SettlementProjection(execution_status="succeeded"),
    parent_materialization=ProjectionInline(
      source=projection.content,
      contract=projection.contract,
      value=projection.inline_view,
    ),
  )


def _events(result: TaskResult, envelope: AgentCompletionEnvelope) -> list[dict]:
  return [
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "task_id": result.attempt.physical_task_id,
      "task_type": "background_agent",
      "started_at": 1.0,
      "metadata": {},
      "capability_bind": {
        "schema_version": "1.0",
        "capability_id": "node.implement",
        "model_key": "test.anthropic.claude-sonnet-4-6",
        "provider": "anthropic",
        "upstream_model": "claude-sonnet-4-6",
        "adapter": "test.anthropic",
        "protocol_profile": "test.reasoning",
        "route": "test.in_process",
        "effort": "high",
        "credential_principal": "user",
        "credential_ref": "test-user:anthropic",
        "run_mode": "interactive",
        "registry_revision": "test-capability-execution.1",
        "policy_revision": "test-capability-execution.1",
        "selection_source": "internal_policy",
      },
    },
    {
      "type": "task_completed",
      "task_id": result.attempt.physical_task_id,
      "final_state": "completed",
      "completed_at": 2.0,
      "result": result.model_dump(mode="json"),
      "error": None,
    },
    build_agent_completion_event(
      task_id=result.attempt.physical_task_id,
      envelope=envelope,
      ts=2.1,
    ),
  ]


def test_registry_replays_exact_task_result_and_completion_idempotently() -> None:
  result = _result()
  envelope = _envelope(result)
  events = _events(result, envelope)
  events.append(dict(events[-1]))

  registry = TaskRegistry()
  registry.load_from_events(events)
  entry = registry.get("bg_7")
  assert entry is not None
  assert entry.state == TaskState.COMPLETED
  assert entry.task_result == result
  assert entry.completion_envelope == envelope


def test_registry_quarantines_conflicting_completion_identity() -> None:
  result = _result()
  events = _events(result, _envelope(result))
  events.append(
    build_agent_completion_event(
      task_id="bg_7",
      envelope=_envelope(result, message_id="conflicting-message"),
      ts=2.2,
    )
  )

  registry = TaskRegistry()
  registry.load_from_events(events)
  entry = registry.get("bg_7")
  assert entry is not None
  assert entry.state == TaskState.FAILED
  assert entry.error is not None
  assert entry.error["code"] == "agent_completion_conflict"


def test_agent_completion_event_rejects_wrong_physical_task() -> None:
  result = _result()
  try:
    build_agent_completion_event(
      task_id="bg_other",
      envelope=_envelope(result),
      ts=1.0,
    )
  except ValueError as exc:
    assert "physical task" in str(exc)
  else:
    raise AssertionError("wrong physical task identity was accepted")


def test_event_adapter_preserves_only_atomic_delivery_envelope() -> None:
  event = {
    "type": "workflow_output_attached",
    "assistant_message_seq": 9,
    "kind": "workflow_primary_output",
    "delivery_envelope": {"schema_version": "1.0"},
    "read": {
      "action": "output",
      "workflow_run_id": "workflow-1",
      "output_id": "output-1",
    },
    "output_id": "legacy-flat-output",
  }
  assert adapt_event(event, 1) == {
    "type": "workflow_output_attached",
    "assistant_message_seq": 9,
    "kind": "workflow_primary_output",
    "delivery_envelope": {"schema_version": "1.0"},
    "read": {
      "action": "output",
      "workflow_run_id": "workflow-1",
      "output_id": "output-1",
    },
  }
