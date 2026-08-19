# ruff: noqa: E402

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


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
from agent_gateway.task_registry import TaskRegistry
from agent_gateway.runner_run_loop import _agent_completion_contract_error
from agent_workflow_contracts import (
  ActivityHandle,
  AgentCompletionEnvelope,
  AgentOperationRef,
  AnalyticalOutcome,
  AttemptRef,
  CanonicalProjection,
  ChildEvidenceProjection,
  ContentHandle,
  ContentReadGrant,
  ContractRef,
  EvidenceObservation,
  ExecutionSettlement,
  NamedArtifact,
  ObservedSourceEvidenceRef,
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


def test_envelope_omits_child_evidence_when_the_child_observed_nothing() -> None:
  """Absent means none: pre-B-4 payloads and digests must replay byte-identically."""

  text = "Exact result"
  result = _task_result(text)
  envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )

  assert envelope.child_evidence is None
  payload = envelope.model_dump(mode="json")
  assert "child_evidence" not in payload
  # A historical row that never carried the field still validates and still
  # dumps to exactly the bytes it was recorded with, so the durable event's
  # digest over it is unchanged.
  replayed = AgentCompletionEnvelope.model_validate(payload)
  assert replayed.model_dump(mode="json") == payload
  assert canonical_json_bytes(replayed.model_dump(mode="json")) == canonical_json_bytes(payload)
  recorded = event_to_dict(AgentCompletionEvent(task_id="bg-1", envelope=replayed, ts=123.5))
  assert recorded["fingerprint"] == event_to_dict(
    AgentCompletionEvent(task_id="bg-1", envelope=envelope, ts=123.5)
  )["fingerprint"]


def test_envelope_carries_the_child_evidence_projection_when_the_child_read() -> None:
  text = "Exact result"
  result_payload = _task_result(text).model_dump(mode="json")
  result_payload["evidence"] = EvidenceObservation(
    observed_sources=(
      ObservedSourceEvidenceRef(
        source_kind="filing",
        document_id="edgar:0000789019-26-000012",
        produced_by_tool="filings_read",
        source_url="https://www.sec.gov/Archives/msft-10k.htm",
      ),
    ),
    tools_used=("filings_read",),
  ).model_dump(mode="json")
  result = TaskResult.model_validate(result_payload)

  envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )

  assert envelope.child_evidence is not None
  assert envelope.child_evidence.evidence_tools == ("filings_read",)
  assert [ref.document_id for ref in envelope.child_evidence.observed_sources] == [
    "edgar:0000789019-26-000012",
  ]
  payload = envelope.model_dump(mode="json")
  assert payload["child_evidence"]["observed_sources"][0]["produced_by_tool"] == "filings_read"
  assert AgentCompletionEnvelope.model_validate(payload) == envelope


def test_child_evidence_projection_must_record_an_observation() -> None:
  with pytest.raises(ValidationError, match="must record an observation"):
    ChildEvidenceProjection()


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


# --- CUR-E2E-08: handle-shaped deliveries must self-describe ---------------


def _handle_shaped_envelope(text: str) -> AgentCompletionEnvelope:
  result = _task_result(text)
  return build_agent_completion_envelope(
    result,
    policy=ParentResultPolicy(
      preferred="terminal_narrative_inline_exact",
      max_inline_bytes=100,
      on_overflow="result_handle",
    ),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )


def test_handle_delivery_notification_summary_carries_dispatch_objective() -> None:
  """A bare result_handle notification names the parent's own dispatch.

  CUR-E2E-08: five identically-named children, a constant summary, and a
  hex task_id gave the parent nothing to map a mid-recovery delivery back
  to its own tracking; the unread result was later called "outstanding".
  """
  envelope = _handle_shaped_envelope("x" * 4_000)
  assert envelope.parent_materialization.kind == "result_handle"

  notification = agent_completion_notification(
    SimpleNamespace(
      task_id="bg-1",
      agent_name="explore",
      notification_generation=1,
      metadata={},
      admitted_task=SimpleNamespace(
        objective="TRACK 3 — VRT margin trend across the last four quarters",
      ),
    ),
    envelope,
    timestamp=123.5,
  )

  assert "TRACK 3 — VRT margin trend" in notification.summary
  assert "get_agent_result_content" in notification.summary
  assert "settled, not running" in notification.summary
  # The identity travels in the summary only: the durable envelope payload
  # is untouched.
  assert notification.payload == envelope.model_dump(mode="json")
  assert notification.inline_payload()[1] is None
  # Bounded under the 2000-char render truncation with room to spare.
  assert len(notification.summary) < 800


def test_handle_delivery_notification_summary_echo_is_bounded() -> None:
  envelope = _handle_shaped_envelope("x" * 4_000)
  notification = agent_completion_notification(
    SimpleNamespace(
      task_id="bg-1",
      agent_name="explore",
      notification_generation=1,
      metadata={"admitted_task": {"objective": "T" * 5_000}},
      admitted_task=None,
    ),
    envelope,
    timestamp=123.5,
  )
  assert "TTTT" in notification.summary
  assert len(notification.summary) < 800


def test_handle_delivery_notification_summary_fails_open_without_objective() -> None:
  envelope = _handle_shaped_envelope("x" * 4_000)
  notification = agent_completion_notification(
    SimpleNamespace(
      task_id="bg-1",
      agent_name="explore",
      notification_generation=1,
      metadata={},
    ),
    envelope,
    timestamp=123.5,
  )
  assert "get_agent_result_content" in notification.summary
  assert "Dispatched objective" not in notification.summary


def test_inline_delivery_notification_summary_is_unchanged() -> None:
  text = "Exact result"
  result = _task_result(text)
  envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )
  assert envelope.parent_materialization.kind == "terminal_narrative_inline_exact"
  notification = agent_completion_notification(
    SimpleNamespace(
      task_id="bg-1",
      agent_name="explore",
      notification_generation=1,
      metadata={},
    ),
    envelope,
    timestamp=123.5,
  )
  assert notification.summary == (
    "Agent completed; consume the typed parent materialization."
  )


# --- CUR-E2E-08: recording that the parent read the delivered handle -------


def test_mark_result_content_read_requires_matching_delivered_handle() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent")
  envelope = _handle_shaped_envelope("x" * 4_000)
  content_id = envelope.parent_materialization.source.content_id

  # No envelope yet: refused.
  assert registry.mark_result_content_read(
    entry.task_id, content_id=content_id
  ) is False

  entry.completion_envelope = envelope
  # Wrong content: refused.
  assert registry.mark_result_content_read(
    entry.task_id, content_id="sha256:" + "0" * 64
  ) is False
  assert entry.result_content_read is False
  # Unknown task: refused.
  assert registry.mark_result_content_read(
    "no-such-task", content_id=content_id
  ) is False
  # Exact delivered handle: recorded, idempotently.
  assert registry.mark_result_content_read(
    entry.task_id, content_id=content_id
  ) is True
  assert entry.result_content_read is True
  assert registry.mark_result_content_read(
    entry.task_id, content_id=content_id
  ) is True


def test_mark_result_content_read_refuses_inline_materialization() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent")
  text = "Exact result"
  result = _task_result(text)
  entry.completion_envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: text,
    read_grant_factory=_read_grant,
  )
  content_id = entry.completion_envelope.parent_materialization.source.content_id
  assert registry.mark_result_content_read(
    entry.task_id, content_id=content_id
  ) is False
  assert entry.result_content_read is False
