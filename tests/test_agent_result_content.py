from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

from agent_gateway.agent_result_content import (
  AGENT_RESULT_CONTENT_MAX_SEQUENCE,
  make_get_agent_result_content_handler,
  make_get_agent_result_content_tool_def,
)
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.final_narrative_artifact import publish_final_narrative
from agent_gateway.runner_session_events import build_agent_completion_event
from agent_gateway.sub_agent_result_contract import (
  canonical_projection,
  report_contract_ref,
  terminal_narrative_content_handle,
)
from agent_workflow_contracts import (
  ActivityHandle,
  AgentCompletionEnvelope,
  AgentOperationRef,
  AttemptRef,
  ContentReadGrant,
  EvidenceObservation,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  ResultHandle,
  SettlementProjection,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultRef,
  TaskResultValues,
  TranscriptHandle,
  UsageObservation,
  canonical_json_bytes,
)


def _run(coro):
  return asyncio.run(coro)


def _digest(value: str) -> str:
  return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_result(*, values: TaskResultValues, task_id: str = "bg-1") -> TaskResult:
  operation = AgentOperationRef(
    namespace="agent-workflow",
    name="research",
    version="v1",
    digest=_digest("operation"),
  )
  return TaskResult(
    task_result_id=f"task-result:{task_id}",
    logical_task=OrdinaryDelegationTaskRef(
      delegation_id=task_id,
      operation=operation,
    ),
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id=f"{task_id}:attempt:1",
      physical_task_id=task_id,
    ),
    execution=ExecutionSettlement(status="succeeded"),
    evidence=EvidenceObservation(),
    values=values,
    observation=TaskObservation(
      transcript=TranscriptHandle(kind="child_transcript", owner_id=task_id),
      activity=ActivityHandle(kind="child_activity", owner_id=task_id),
      usage=UsageObservation(),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=_digest("admitted"),
      model_bind_digest=_digest("model"),
      capability_binding_digest=_digest("capability"),
      tool_grant_digest=_digest("tools"),
    ),
  )


def _envelope(result: TaskResult, *, principal_id: str = "runner-1") -> AgentCompletionEnvelope:
  source = (
    result.values.terminal_narrative
    or (result.values.projection.content if result.values.projection is not None else None)
  )
  assert source is not None
  return AgentCompletionEnvelope(
    message_id=f"completion:{result.attempt.physical_task_id}",
    task_result_ref=TaskResultRef.from_result(result),
    settlement_projection=SettlementProjection(execution_status="succeeded"),
    parent_materialization=ResultHandle(
      source=source,
      read_grant=ContentReadGrant(
        grant_id=f"grant:{result.attempt.physical_task_id}",
        content_id=source.content_id,
        scope="direct_parent",
        principal_id=principal_id,
      ),
    ),
  )


async def _append_completion(
  log: AgentSessionLog,
  result: TaskResult,
  envelope: AgentCompletionEnvelope,
  *,
  session_id: str = "session-1",
  workflow_owned: bool = False,
) -> None:
  task_id = result.attempt.physical_task_id
  await log.append({
    "type": "task_registered",
    "task_id": task_id,
    "parent_session_id": session_id,
    "metadata": (
      {"workflow_task": {"workflow_run_id": "wf-1"}}
      if workflow_owned
      else {}
    ),
  })
  await log.append({
    "type": "task_completed",
    "task_id": task_id,
    "result": result.model_dump(mode="json"),
    "error": None,
  })
  await log.append(build_agent_completion_event(
    task_id=task_id,
    envelope=envelope,
    ts=3.0,
  ))


def _runner(log: AgentSessionLog, workspace: Path):
  return SimpleNamespace(
    _agent_session_log=log,
    _workspace_dir=workspace,
    _gateway_session_id="session-1",
    _runner_id="runner-1",
  )


def test_tool_is_generic_result_content_reader() -> None:
  definition = make_get_agent_result_content_tool_def()
  assert definition["name"] == "get_agent_result_content"
  assert set(definition["input_schema"]["required"]) == {
    "content_id",
    "read_grant_id",
  }


def test_terminal_narrative_pages_reassemble_exact_utf8(tmp_path: Path) -> None:
  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = publish_final_narrative(
      workspace_dir=workspace,
      sub_agent_id="sub:1",
      terminal_event_seq=7,
      text=("alpha🙂\n" * 8_000) + "terminal",
    )
    result = _task_result(values=TaskResultValues(
      terminal_narrative=terminal_narrative_content_handle(reference),
    ))
    envelope = _envelope(result)
    log = AgentSessionLog(tmp_path / "session.jsonl")
    await _append_completion(log, result, envelope)
    handler = make_get_agent_result_content_handler([_runner(log, workspace)])

    chunks: list[str] = []
    after_char = 0
    while True:
      page, error = await handler({
        "content_id": envelope.parent_materialization.source.content_id,
        "read_grant_id": envelope.parent_materialization.read_grant.grant_id,
        "after_char": after_char,
      })
      assert error is None
      assert page is not None
      assert page["kind"] == "agent_result_content"
      assert page["value_kind"] == "terminal_narrative"
      chunks.append(page["content"])
      if page["end"]:
        break
      after_char = page["next_cursor"]["after_char"]
    assert "".join(chunks) == ("alpha🙂\n" * 8_000) + "terminal"

  _run(_case())


def test_projection_json_pages_reassemble_canonical_bytes(tmp_path: Path) -> None:
  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    value = {"coverage": [f"row-{index}" for index in range(4_000)]}
    projection = canonical_projection(
      contract=report_contract_ref("explore-findings-v1"),
      value=value,
    )
    result = _task_result(values=TaskResultValues(projection=projection))
    envelope = _envelope(result)
    log = AgentSessionLog(tmp_path / "session.jsonl")
    await _append_completion(log, result, envelope)
    handler = make_get_agent_result_content_handler([_runner(log, workspace)])

    chunks: list[str] = []
    after_char = 0
    while True:
      page, error = await handler({
        "content_id": projection.content.content_id,
        "read_grant_id": envelope.parent_materialization.read_grant.grant_id,
        "after_char": after_char,
      })
      assert error is None
      assert page is not None
      assert page["value_kind"] == "projection"
      assert page["media_type"] == "application/json"
      chunks.append(page["content"])
      if page["end"]:
        break
      after_char = page["next_cursor"]["after_char"]
    assert "".join(chunks).encode("utf-8") == canonical_json_bytes(value)

  _run(_case())


def test_read_requires_exact_principal_and_ordinary_registration(tmp_path: Path) -> None:
  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    projection = canonical_projection(
      contract=report_contract_ref("explore-findings-v1"),
      value={"answer": "exact"},
    )
    result = _task_result(values=TaskResultValues(projection=projection))
    envelope = _envelope(result, principal_id="different-runner")
    log = AgentSessionLog(tmp_path / "session.jsonl")
    await _append_completion(log, result, envelope)
    handler = make_get_agent_result_content_handler([_runner(log, workspace)])
    page, error = await handler({
      "content_id": projection.content.content_id,
      "read_grant_id": envelope.parent_materialization.read_grant.grant_id,
    })
    assert page is None
    assert error is not None
    assert error["code"] == "result_content_unavailable"

    workflow_log = AgentSessionLog(tmp_path / "workflow-session.jsonl")
    workflow_envelope = _envelope(result)
    await _append_completion(
      workflow_log,
      result,
      workflow_envelope,
      workflow_owned=True,
    )
    workflow_handler = make_get_agent_result_content_handler([
      _runner(workflow_log, workspace)
    ])
    page, error = await workflow_handler({
      "content_id": projection.content.content_id,
      "read_grant_id": workflow_envelope.parent_materialization.read_grant.grant_id,
    })
    assert page is None
    assert error is not None
    assert error["code"] == "result_content_unavailable"

  _run(_case())


def test_invalid_cursor_is_rejected_before_content_read(tmp_path: Path) -> None:
  handler = make_get_agent_result_content_handler([
    _runner(AgentSessionLog(tmp_path / "session.jsonl"), tmp_path)
  ])
  page, error = _run(handler({
    "content_id": "sha256:" + ("0" * 64),
    "read_grant_id": "grant-1",
    "after_char": AGENT_RESULT_CONTENT_MAX_SEQUENCE + 1,
  }))
  assert page is None
  assert error is not None
  assert error["code"] == "invalid_input"
