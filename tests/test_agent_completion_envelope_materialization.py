# ruff: noqa: E402

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.agent_session_log import (
  AgentSessionLog,
  IdempotentEventConflictError,
)
from agent_gateway.events import AgentCompletionEvent, event_from_dict, event_to_dict
from agent_gateway.runner_background_tasks import (
  DEFAULT_PARENT_RESULT_MAX_INLINE_BYTES,
  ParentResultMaterializationError,
  agent_completion_notification,
  agent_completion_message_id,
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
)
from agent_gateway.runner_run_loop import _agent_completion_contract_error
from agent_workflow_contracts import (
  ActivityHandle,
  AgentOperationRef,
  AnalyticalOutcome,
  AttemptRef,
  CanonicalProjection,
  ContentHandle,
  ContentReadGrant,
  ContractRef,
  EvidenceObservation,
  ExecutionSettlement,
  NamedArtifact,
  OrdinaryDelegationTaskRef,
  ParentResultPolicy,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  TranscriptHandle,
  UsageObservation,
  WorkflowNodeTaskRef,
  canonical_json_bytes,
)


def _digest(value: str) -> str:
  return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contract(name: str) -> ContractRef:
  return ContractRef(
    namespace="agent-workflow",
    name=name,
    version="v1",
    digest=_digest(f"contract:{name}"),
  )


def _operation() -> AgentOperationRef:
  return AgentOperationRef(
    namespace="agent-workflow",
    name="research",
    version="v1",
    digest=_digest("operation:research"),
  )


def _text_handle(text: str) -> ContentHandle:
  encoded = text.encode("utf-8")
  digest = hashlib.sha256(encoded).hexdigest()
  return ContentHandle(
    content_id=f"sha256:{digest}",
    content_sha256=digest,
    content_bytes=len(encoded),
    content_chars=len(text),
    contract=_contract("terminal-narrative"),
    media_type="text/markdown; charset=utf-8",
    encoding="utf-8",
    retention="durable",
  )


def _ordinary_task() -> OrdinaryDelegationTaskRef:
  return OrdinaryDelegationTaskRef(
    delegation_id="delegation-1",
    operation=_operation(),
  )


def _attempt() -> AttemptRef:
  return AttemptRef(
    attempt_number=1,
    attempt_id="attempt-1",
    physical_task_id="bg-1",
  )


def _task_result(
  text: str,
  *,
  logical_task: OrdinaryDelegationTaskRef | WorkflowNodeTaskRef | None = None,
  projection: CanonicalProjection | None = None,
) -> TaskResult:
  narrative = _text_handle(text)
  return TaskResult(
    task_result_id="task-result-1",
    logical_task=logical_task or _ordinary_task(),
    attempt=_attempt(),
    execution=ExecutionSettlement(status="succeeded"),
    outcome=AnalyticalOutcome(
      disposition="complete",
      assessment_source="domain_tool",
    ),
    evidence=EvidenceObservation(),
    values=TaskResultValues(
      terminal_narrative=narrative,
      projection=projection,
    ),
    observation=TaskObservation(
      transcript=TranscriptHandle(
        kind="child_transcript",
        owner_id="bg-1",
      ),
      activity=ActivityHandle(
        kind="child_activity",
        owner_id="bg-1",
      ),
      usage=UsageObservation(output_tokens=10),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=_digest("admitted"),
      model_bind_digest=_digest("model"),
      capability_binding_digest=_digest("capability"),
      tool_grant_digest=_digest("tools"),
    ),
  )


def _read_grant(source: ContentHandle) -> ContentReadGrant:
  return ContentReadGrant(
    grant_id=f"grant:{source.content_sha256}",
    content_id=source.content_id,
    scope="direct_parent",
    principal_id="parent-1",
  )


def test_direct_narrative_defaults_to_exact_terminal_message() -> None:
  text = "Complete child report, with no clipping."
  result = _task_result(text)
  policy = ordinary_parent_result_policy(result)

  envelope = build_agent_completion_envelope(
    result,
    policy=policy,
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )

  assert policy == ParentResultPolicy(
    preferred="terminal_narrative_inline_exact",
    max_inline_bytes=DEFAULT_PARENT_RESULT_MAX_INLINE_BYTES,
    on_overflow="result_handle",
  )
  assert envelope.parent_materialization.kind == "terminal_narrative_inline_exact"
  assert envelope.parent_materialization.content == text
  assert envelope.parent_materialization.complete is True
  assert envelope.task_result_ref.task_result_id == result.task_result_id


def test_oversized_narrative_delivers_handle_without_reading_or_preview() -> None:
  text = "x" * 10_000
  result = _task_result(text)
  reader_called = False

  def reader(_result: TaskResult) -> str:
    nonlocal reader_called
    reader_called = True
    return text

  envelope = build_agent_completion_envelope(
    result,
    policy=ParentResultPolicy(
      preferred="terminal_narrative_inline_exact",
      max_inline_bytes=100,
      on_overflow="result_handle",
    ),
    terminal_narrative_reader=reader,
    read_grant_factory=_read_grant,
  )

  assert reader_called is False
  assert envelope.parent_materialization.kind == "result_handle"
  assert envelope.parent_materialization.source.content_chars == len(text)
  wire = envelope.model_dump(mode="json")
  assert "x" * 101 not in str(wire)
  assert "text_truncated" not in str(wire)


def test_overflow_uses_only_explicit_operation_authored_summary() -> None:
  text = "full canonical answer" * 100
  result = _task_result(text)
  envelope = build_agent_completion_envelope(
    result,
    policy=ParentResultPolicy(
      preferred="terminal_narrative_inline_exact",
      max_inline_bytes=64,
      on_overflow="authored_summary_with_result_handle",
    ),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
    authored_summary="Authored synthesis.",
  )
  assert envelope.parent_materialization.kind == "authored_summary_with_result_handle"
  assert envelope.parent_materialization.summary == "Authored synthesis."

  with pytest.raises(
    ParentResultMaterializationError,
    match="operation-authored summary",
  ):
    build_agent_completion_envelope(
      result,
      policy=ParentResultPolicy(
        preferred="terminal_narrative_inline_exact",
        max_inline_bytes=64,
        on_overflow="authored_summary_with_result_handle",
      ),
      terminal_narrative_reader=lambda _result: text,
      read_grant_factory=_read_grant,
    )


def test_projection_policy_preserves_exact_typed_value() -> None:
  value = {"coverage": "complete", "count": 2}
  raw = canonical_json_bytes(value)
  digest = hashlib.sha256(raw).hexdigest()
  contract = _contract("research-projection")
  projection = CanonicalProjection(
    contract=contract,
    content=ContentHandle(
      content_id=f"sha256:{digest}",
      content_sha256=digest,
      content_bytes=len(raw),
      content_chars=len(raw.decode("utf-8")),
      contract=contract,
      media_type="application/json",
      encoding="utf-8",
      retention="durable",
    ),
    inline_view=value,
  )
  result = _task_result("narrative remains canonical", projection=projection)

  envelope = build_agent_completion_envelope(
    result,
    policy=ParentResultPolicy(
      preferred="projection_inline",
      max_inline_bytes=1_000,
      on_overflow="fail",
    ),
    terminal_narrative_reader=lambda _result: "unused",
    read_grant_factory=_read_grant,
  )

  assert envelope.parent_materialization.kind == "projection_inline"
  assert envelope.parent_materialization.value == value


def test_completion_identity_is_stable_and_event_round_trips() -> None:
  text = "Exact result"
  result = _task_result(text)
  envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )
  assert envelope.message_id == agent_completion_message_id(result)
  assert agent_completion_message_id(result) == agent_completion_message_id(result)

  event = AgentCompletionEvent(task_id="bg-1", envelope=envelope, ts=123.5)
  payload = event_to_dict(event)
  assert event_from_dict(payload) == event
  assert payload["event_id"] == event.event_id
  assert payload["fingerprint"] == event.fingerprint
  assert payload["envelope"]["message_id"] == envelope.message_id
  with pytest.raises(ValueError, match="fingerprint conflicts"):
    event_from_dict({**payload, "fingerprint": _digest("conflict")})
  with pytest.raises(ValueError, match="event_id conflicts"):
    event_from_dict({key: value for key, value in payload.items() if key != "event_id"})

  notification = agent_completion_notification(
    SimpleNamespace(
      task_id="bg-1",
      agent_name="researcher",
      notification_generation=1,
      metadata={},
    ),
    envelope,
    timestamp=123.5,
  )
  assert notification.inline_payload()[1] is None
  assert notification.payload == envelope.model_dump(mode="json")


@pytest.mark.asyncio
async def test_agent_completion_append_is_atomic_idempotent_and_fail_closed(
  tmp_path: Path,
) -> None:
  text = "Exact result"
  result = _task_result(text)
  envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )
  first = event_to_dict(AgentCompletionEvent(
    task_id="bg-1",
    envelope=envelope,
    ts=123.5,
  ))
  replay = event_to_dict(AgentCompletionEvent(
    task_id="bg-1",
    envelope=envelope,
    ts=124.5,
  ))
  session_path = tmp_path / "session.jsonl"
  session_log = AgentSessionLog(session_path)
  competing_writer = AgentSessionLog(session_path)

  accepted = await asyncio.gather(
    session_log.append(first),
    competing_writer.append(replay),
  )
  assert [entry.seq for entry in accepted] == [1, 1]
  stored, _ = await session_log.query(event_types={"agent_completion"})
  assert len(stored) == 1

  other_text = "Conflicting result"
  other_result = _task_result(other_text)
  other_envelope = build_agent_completion_envelope(
    other_result,
    policy=ordinary_parent_result_policy(other_result),
    terminal_narrative_reader=lambda _result: other_text,
    read_grant_factory=_read_grant,
  )
  conflict = event_to_dict(AgentCompletionEvent(
    task_id="bg-1",
    envelope=other_envelope,
    ts=125.5,
  ))
  assert conflict["event_id"] == first["event_id"]
  assert conflict["fingerprint"] != first["fingerprint"]
  with pytest.raises(IdempotentEventConflictError, match="conflicts"):
    await session_log.append(conflict)


def test_workflow_node_result_cannot_enter_direct_parent_completion() -> None:
  workflow_task = WorkflowNodeTaskRef(
    workflow_run_id="workflow-1",
    plan_id="plan-1",
    phase_number=1,
    revision=1,
    node_id="research",
    operation=_operation(),
  )
  result = _task_result("workflow child report", logical_task=workflow_task)

  with pytest.raises(
    ParentResultMaterializationError,
    match="workflow aggregation",
  ):
    build_agent_completion_envelope(
      result,
      policy=ParentResultPolicy(
        preferred="terminal_narrative_inline_exact",
        max_inline_bytes=1_000,
        on_overflow="result_handle",
      ),
      terminal_narrative_reader=lambda _result: "workflow child report",
      read_grant_factory=_read_grant,
    )


def test_artifact_only_result_does_not_emit_an_unreadable_handle() -> None:
  result_payload = _task_result("placeholder").model_dump(mode="json")
  artifact = _text_handle("artifact bytes")
  result_payload["values"] = {
    "terminal_narrative": None,
    "projection": None,
    "artifacts": [
      NamedArtifact(name="primary", content=artifact).model_dump(mode="json")
    ],
  }
  result = TaskResult.model_validate(result_payload)

  with pytest.raises(
    ParentResultMaterializationError,
    match="artifact-only result content has no registered direct-parent reader",
  ):
    build_agent_completion_envelope(
      result,
      policy=ParentResultPolicy(
        preferred="result_handle",
        max_inline_bytes=1,
        on_overflow="result_handle",
      ),
      terminal_narrative_reader=lambda _result: "unused",
      read_grant_factory=_read_grant,
    )


def test_run_loop_rejects_a_clipped_normalized_completion_envelope() -> None:
  assert _agent_completion_contract_error(
    tool_uses=[(
      "tool-1",
      "run_agent",
      {"task": "research", "background": False},
    )],
    tool_result_blocks=[{
      "type": "tool_result",
      "tool_use_id": "tool-1",
      "content": (
        '{"schema_version":"1.0","message_id":"agent-completion:'
        + "a" * 64
        + '","task_result_ref":...<truncated>'
      ),
    }],
  ) == (
    "agent_completion_contract_invalid: normalized run_agent result was not "
    "a complete AgentCompletionEnvelope after context materialization "
    "(JSONDecodeError)."
  )


def test_run_loop_accepts_a_complete_normalized_completion_envelope() -> None:
  text = "Exact result"
  result = _task_result(text)
  envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )
  assert _agent_completion_contract_error(
    tool_uses=[(
      "tool-1",
      "run_agent",
      {"task": "research", "background": False},
    )],
    tool_result_blocks=[{
      "type": "tool_result",
      "tool_use_id": "tool-1",
      "content": envelope.model_dump_json(),
    }],
  ) is None
