from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from agent_gateway.admission import ParentContext
from agent_gateway.agent_result_content import (
  AGENT_RESULT_CONTENT_MAX_SEQUENCE,
  make_get_agent_result_content_handler,
)
from agent_gateway.approval_route import NoApprovalRoute
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.task_registry import TaskEntry, TaskRegistry, TaskState
from agent_gateway.final_narrative_artifact import (
  publish_final_narrative,
  read_final_narrative_by_content_handle,
)
from agent_gateway.runner_background_tasks import (
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
  task_completed_event_payload,
  task_correlation_payload,
  task_registered_event_payload,
)
from agent_gateway.runner_session_events import build_agent_completion_event
from agent_gateway.sub_agent_result_contract import (
  canonical_projection,
  build_task_result,
  report_contract_ref,
  terminal_narrative_content_handle,
)
from agent_workflow_contracts import (
  AdmittedTask,
  ActivityHandle,
  AgentCompletionEnvelope,
  AgentOperationRef,
  AuthoredSummaryWithResultHandle,
  AttemptRef,
  ContentHandle,
  ContentReadGrant,
  EvidenceObservation,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  ParentResultPolicy,
  ProjectionInline,
  ResultHandle,
  SettlementProjection,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultRef,
  TaskResultValues,
  TerminalNarrativeInlineExact,
  TranscriptHandle,
  UsageObservation,
  canonical_json_bytes,
)
from agent_workflow_contracts.models import (
  AdmittedResearchBinding,
  AdmittedTenant,
)
from gateway_test_support.admitted_authority_test_support import provenance_of
from gateway_test_support.resumer_custody_test_support import canonical_read_custody_task


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


def _envelope(
  result: TaskResult,
  *,
  principal_id: str = "runner-1",
) -> tuple[AgentCompletionEnvelope, ResultHandle]:
  source = (
    result.values.terminal_narrative
    or (result.values.projection.content if result.values.projection is not None else None)
  )
  assert source is not None
  materialization = ResultHandle(
    source=source,
    read_grant=ContentReadGrant(
      grant_id=f"grant:{result.attempt.physical_task_id}",
      content_id=source.content_id,
      scope="direct_parent",
      principal_id=principal_id,
    ),
  )
  return (
    AgentCompletionEnvelope(
      message_id=f"completion:{result.attempt.physical_task_id}",
      task_result_ref=TaskResultRef.from_result(result),
      settlement_projection=SettlementProjection(execution_status="succeeded"),
      parent_materialization=materialization,
    ),
    materialization,
  )


async def _append_completion(
  log: AgentSessionLog,
  result: TaskResult,
  envelope: AgentCompletionEnvelope,
  *,
  session_id: str = "session-1",
  parent_job_id: str = "session-1",
  workflow_owned: bool = False,
) -> None:
  task_id = result.attempt.physical_task_id
  metadata: dict[str, object] = {
    "admitted_task": {
      "execution_snapshot": {
        "admitted_context": {"lineage": {"parent_job_id": parent_job_id}},
      },
    },
  }
  if workflow_owned:
    metadata["workflow_task"] = {"workflow_run_id": "wf-1"}
  await log.append({
    "type": "task_registered",
    "task_id": task_id,
    "parent_session_id": session_id,
    "metadata": metadata,
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
  )


def _job(job_id: str, *, handle_grants: frozenset[str] = frozenset()) -> ParentContext:
  return ParentContext(
    tenant=AdmittedTenant(tenant_id=None, storage_id="alice", role="owner"),
    research_binding=AdmittedResearchBinding(),
    children=None,
    tool_ids=None,
    withheld=frozenset(),
    handle_grants=handle_grants,
    remaining_usd=None,
    job_id=job_id,
    depth=0,
    approval_route=NoApprovalRoute(),
  )


def _root() -> ParentContext:
  return _job("session-1")


def _producer_envelope(
  result: TaskResult,
  workspace: Path,
  *,
  policy: ParentResultPolicy | None = None,
) -> AgentCompletionEnvelope:
  def _grant(source: ContentHandle) -> ContentReadGrant:
    return ContentReadGrant(
      grant_id=f"grant:{result.attempt.physical_task_id}",
      content_id=source.content_id,
      scope="direct_parent",
      principal_id="runner-1",
    )

  def _narrative(canonical: TaskResult) -> str:
    source = canonical.values.terminal_narrative
    assert source is not None
    return read_final_narrative_by_content_handle(
      workspace_dir=workspace,
      content=source,
    )

  return build_agent_completion_envelope(
    result,
    policy=policy or ordinary_parent_result_policy(result),
    terminal_narrative_reader=_narrative,
    read_grant_factory=_grant,
    authored_summary="Operation-authored summary",
  )


@pytest.mark.parametrize("value_kind", ["terminal_narrative", "projection"])
def test_real_inline_producer_content_is_readable_without_grant(
  tmp_path: Path,
  value_kind: str,
) -> None:
  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = None
    if value_kind == "terminal_narrative":
      text = "Short published ordinary narrative 𐍈\n"
      reference = publish_final_narrative(
        workspace_dir=workspace,
        sub_agent_id="sub:inline",
        terminal_event_seq=7,
        text=text,
      )
      values = TaskResultValues(
        terminal_narrative=terminal_narrative_content_handle(reference),
      )
    else:
      value = {"z": "exact 𐍈", "a": [1, 2]}
      text = canonical_json_bytes(value).decode("utf-8")
      values = TaskResultValues(projection=canonical_projection(
        contract=report_contract_ref("explore-findings-v1"),
        value=value,
      ))
    result = _task_result(values=values)
    envelope = _producer_envelope(result, workspace)
    materialization = envelope.parent_materialization
    assert isinstance(materialization, (TerminalNarrativeInlineExact, ProjectionInline))
    content_id = materialization.source.content_id
    log = AgentSessionLog(tmp_path / "session.jsonl")
    await _append_completion(log, result, envelope)

    async def _read(reader: ParentContext, **selector):
      handler = make_get_agent_result_content_handler(
        [_runner(log, workspace)],
        caller=lambda: reader,
      )
      return await handler({"content_id": content_id, **selector})

    for reader in (_root(), _job("sibling", handle_grants=frozenset({content_id}))):
      page, error = await _read(reader)
      assert error is None and page is not None
      assert page["value_kind"] == value_kind
      assert page["content"].encode("utf-8") == text.encode("utf-8")
      assert page["end"] is True
    page, error = await _read(_job("sibling"))
    assert page is None and error is not None
    assert error["code"] == "handle_not_granted"
    # An explicit grant selects a grant-bearing materialization, not an
    # inline source and not a newly minted authority.
    page, error = await _read(_root(), read_grant_id="grant:bg-1")
    assert page is None and error is not None
    assert error["code"] == "result_content_unavailable"
    if value_kind == "terminal_narrative":
      assert reference is not None
      payload = (workspace / reference.artifact_ref).parent / "payload.txt"
      payload.write_text(text.replace("Short", "Other"), encoding="utf-8")
      page, error = await _read(_root())
      assert page is None and error is not None
      assert error["code"] == "result_content_unavailable"
    else:
      # Corrupt canonical projection bytes without changing the advertised
      # source/ref. Durable result verification must still reject the read.
      projection = result.values.projection
      assert projection is not None
      corrupted = result.model_copy(update={
        "values": result.values.model_copy(update={
          "projection": projection.model_copy(update={
            "inline_view": {"answer": "not the published canonical bytes"},
          }),
        }),
      })
      corrupt_log = AgentSessionLog(tmp_path / "corrupt-session.jsonl")
      await _append_completion(corrupt_log, corrupted, envelope)
      corrupt_handler = make_get_agent_result_content_handler(
        [_runner(corrupt_log, workspace)],
        caller=_root,
      )
      page, error = await corrupt_handler({"content_id": content_id})
      assert page is None and error is not None
      assert error["code"] == "result_content_unavailable"

  _run(_case())


@pytest.mark.parametrize("delivery", ["result_handle", "authored_summary_with_result_handle"])
def test_real_producer_grant_selector_distinguishes_identical_inline_content(
  tmp_path: Path,
  delivery: Literal["result_handle", "authored_summary_with_result_handle"],
) -> None:
  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    projection = canonical_projection(
      contract=report_contract_ref("explore-findings-v1"),
      value={"answer": "identical canonical bytes"},
    )
    inline_result = _task_result(
      values=TaskResultValues(projection=projection),
      task_id="inline",
    )
    handle_result = _task_result(
      values=TaskResultValues(projection=projection),
      task_id="handled",
    )
    inline_envelope = _producer_envelope(inline_result, workspace)
    handle_envelope = _producer_envelope(
      handle_result,
      workspace,
      policy=ParentResultPolicy(
        preferred="projection_inline",
        max_inline_bytes=1,
        on_overflow=delivery,
      ),
    )
    materialization = handle_envelope.parent_materialization
    assert isinstance(materialization, (ResultHandle, AuthoredSummaryWithResultHandle))
    assert materialization.kind == delivery
    log = AgentSessionLog(tmp_path / "session.jsonl")
    await _append_completion(log, inline_result, inline_envelope)
    await _append_completion(log, handle_result, handle_envelope)
    handler = make_get_agent_result_content_handler(
      [_runner(log, workspace)],
      caller=_root,
    )
    for selector in ({}, {"read_grant_id": materialization.read_grant.grant_id}):
      page, error = await handler({
        "content_id": projection.content.content_id,
        **selector,
      })
      assert error is None and page is not None
      assert page["content"].encode("utf-8") == canonical_json_bytes(projection.inline_view)
    page, error = await handler({
      "content_id": projection.content.content_id,
      "read_grant_id": "grant:inline",
    })
    assert page is None and error is not None
    assert error["code"] == "result_content_unavailable"

  _run(_case())


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
    envelope, materialization = _envelope(result)
    log = AgentSessionLog(tmp_path / "session.jsonl")
    await _append_completion(log, result, envelope)
    handler = make_get_agent_result_content_handler([_runner(log, workspace)], caller=_root)

    chunks: list[str] = []
    after_char = 0
    while True:
      page, error = await handler({
        "content_id": materialization.source.content_id,
        "read_grant_id": materialization.read_grant.grant_id,
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
    envelope, materialization = _envelope(result)
    log = AgentSessionLog(tmp_path / "session.jsonl")
    await _append_completion(log, result, envelope)
    handler = make_get_agent_result_content_handler([_runner(log, workspace)], caller=_root)

    chunks: list[str] = []
    after_char = 0
    while True:
      page, error = await handler({
        "content_id": projection.content.content_id,
        "read_grant_id": materialization.read_grant.grant_id,
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


def test_read_is_admitted_for_ancestors_and_passed_handles_only(tmp_path: Path) -> None:
  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log = AgentSessionLog(tmp_path / "session.jsonl")
    lead_result = _task_result(
      values=TaskResultValues(projection=canonical_projection(
        contract=report_contract_ref("explore-findings-v1"),
        value={"answer": "lead"},
      )),
      task_id="bg-1",
    )
    lead_envelope, _ = _envelope(lead_result)
    await _append_completion(log, lead_result, lead_envelope)
    worker_projection = canonical_projection(
      contract=report_contract_ref("explore-findings-v1"),
      value={"answer": "worker"},
    )
    worker_result = _task_result(
      values=TaskResultValues(projection=worker_projection),
      task_id="bg-1.w",
    )
    worker_envelope, _ = _envelope(worker_result, principal_id="lead-runner")
    await _append_completion(log, worker_result, worker_envelope, parent_job_id="bg-1")
    content_id = worker_projection.content.content_id

    async def _read(reader: ParentContext):
      handler = make_get_agent_result_content_handler(
        [_runner(log, workspace)],
        caller=lambda: reader,
      )
      return await handler({"content_id": content_id})

    # The worker's parent and the root (its ancestor) read it.
    for ancestor in ("bg-1", "session-1"):
      page, error = await _read(_job(ancestor))
      assert error is None and page is not None
      assert page["content"] == '{"answer":"worker"}'
    # A sibling of the lead holds no grant; another tree's root neither.
    for stranger in ("bg-2", "session-2"):
      page, error = await _read(_job(stranger))
      assert page is None and error is not None
      assert error["code"] == "handle_not_granted"
    # A sibling that was passed the handle at spawn reads it.
    page, error = await _read(_job("bg-2", handle_grants=frozenset({content_id})))
    assert error is None and page is not None

  _run(_case())


def test_workflow_owned_result_is_not_an_ordinary_materialization(tmp_path: Path) -> None:
  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    projection = canonical_projection(
      contract=report_contract_ref("explore-findings-v1"),
      value={"answer": "exact"},
    )
    result = _task_result(values=TaskResultValues(projection=projection))
    workflow_log = AgentSessionLog(tmp_path / "workflow-session.jsonl")
    workflow_envelope, _ = _envelope(result)
    await _append_completion(
      workflow_log,
      result,
      workflow_envelope,
      workflow_owned=True,
    )
    handler = make_get_agent_result_content_handler(
      [_runner(workflow_log, workspace)],
      caller=_root,
    )
    page, error = await handler({"content_id": projection.content.content_id})
    assert page is None
    assert error is not None
    assert error["code"] == "result_content_unavailable"

  _run(_case())


def test_invalid_cursor_is_rejected_before_content_read(tmp_path: Path) -> None:
  handler = make_get_agent_result_content_handler(
    [_runner(AgentSessionLog(tmp_path / "session.jsonl"), tmp_path)],
    caller=_root,
  )
  page, error = _run(handler({
    "content_id": "sha256:" + ("0" * 64),
    "read_grant_id": "grant-1",
    "after_char": AGENT_RESULT_CONTENT_MAX_SEQUENCE + 1,
  }))
  assert page is None
  assert error is not None
  assert error["code"] == "invalid_input"


def test_successful_read_marks_registry_result_content_read(tmp_path: Path) -> None:
  """The handler records grant redemption on the live registry (CUR-E2E-08).

  End-to-end through the real handler: without this seam, the
  natural-finish reminder fires spuriously for results the parent DID
  read. The mark is best-effort — a registry-less runner (the other tests
  here) must keep working, and an entry-less registry must not error.
  """

  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = publish_final_narrative(
      workspace_dir=workspace,
      sub_agent_id="sub:1",
      terminal_event_seq=7,
      text="## TRACK 3\n" + ("margin analysis " * 400),
    )
    result = _task_result(values=TaskResultValues(
      terminal_narrative=terminal_narrative_content_handle(reference),
    ))
    envelope, materialization = _envelope(result)
    log = AgentSessionLog(tmp_path / "session.jsonl")
    await _append_completion(log, result, envelope)

    registry = TaskRegistry()
    entry = registry.register(
      "background_agent",
      task_id=result.attempt.physical_task_id,
    )
    entry.completion_envelope = envelope
    runner = _runner(log, workspace)
    runner._task_registry = registry
    handler = make_get_agent_result_content_handler([runner], caller=_root)

    assert entry.result_content_read is False
    page, error = await handler({
      "content_id": materialization.source.content_id,
      "read_grant_id": materialization.read_grant.grant_id,
    })
    assert error is None
    assert page is not None
    assert entry.result_content_read is True

    # Entry-less registry: the read itself must stay unaffected.
    bare_runner = _runner(log, workspace)
    bare_runner._task_registry = TaskRegistry()
    bare_handler = make_get_agent_result_content_handler([bare_runner], caller=_root)
    page, error = await bare_handler({
      "content_id": materialization.source.content_id,
      "read_grant_id": materialization.read_grant.grant_id,
    })
    assert error is None
    assert page is not None

  _run(_case())


def test_canonical_continuation_actor_reads_exact_retained_result_content(tmp_path: Path) -> None:
  async def _case() -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    durable_log = AgentSessionLog(tmp_path / "custody.jsonl")

    async def _publish(admitted: AdmittedTask) -> str:
      task_id = admitted.attempt.physical_task_id
      projection = canonical_projection(
        contract=report_contract_ref("explore-findings-v1"),
        value={"answer": task_id},
      )
      result = build_task_result(
        logical_task=admitted.logical_task,
        attempt=admitted.attempt,
        requirement=admitted.result_requirement,
        provenance=provenance_of(admitted),
        execution=ExecutionSettlement(status="succeeded"),
        outcome=None,
        terminal_narrative=None,
        projection=None,
        runtime_projection=projection,
      )
      envelope = _producer_envelope(result, workspace)
      assert admitted.model_bind is not None
      entry = TaskEntry(
        task_id=task_id,
        task_type="background_agent",
        started_at=1.0,
        completed_at=2.0,
        state=TaskState.COMPLETED,
        admitted_task=admitted,
        task_result=result,
        completion_envelope=envelope,
        capability_bind_receipt=admitted.model_bind.model_dump(mode="json"),
        result=result.model_dump(mode="json"),
        metadata={"admitted_task": admitted.model_dump(mode="json")},
        original_task_id=admitted.attempt.resume_of_task_id,
      )
      correlation = task_correlation_payload(entry, runner_id="runner-alias", role="owner")
      await durable_log.append(task_registered_event_payload(
        entry,
        correlation_payload=correlation,
        agent_name="research",
        parent_session_id="session-alias",
      ))
      await durable_log.append(task_completed_event_payload(
        entry,
        TaskState.COMPLETED,
        correlation_payload=correlation,
        completed_at=2.0,
      ))
      await durable_log.append(build_agent_completion_event(
        task_id=task_id,
        envelope=envelope,
        ts=2.0,
      ))
      return projection.content.content_id

    a0 = canonical_read_custody_task("a0", parent_job_id="session-1")
    a1 = canonical_read_custody_task("a1", predecessor=a0, resumer_job_id="nested-actor")
    a2 = canonical_read_custody_task("a2", predecessor=a1, resumer_job_id="later-actor")
    tasks = (
      a0,
      canonical_read_custody_task("old-child", parent_job_id="a0"),
      canonical_read_custody_task("old-sibling", parent_job_id="session-1"),
      canonical_read_custody_task("nested-actor", parent_job_id="actor-parent"),
      a1, canonical_read_custody_task("new-child", parent_job_id="a1"),
      a2, canonical_read_custody_task("latest-child", parent_job_id="a2"),
      canonical_read_custody_task("unrelated", parent_job_id="other-root"),
    )
    content_ids = {}
    for admitted in tasks:
      content_ids[admitted.attempt.physical_task_id] = await _publish(admitted)

    # New runner, empty registry, same durable authority and exact producer bytes.
    replay = AgentSessionLog(tmp_path / "custody.jsonl")
    runner = _runner(replay, workspace)
    runner._task_registry = TaskRegistry()

    async def _read(reader: str, task_id: str):
      handler = make_get_agent_result_content_handler(
        [runner],
        caller=lambda: _job(reader),
      )
      return await handler({"content_id": content_ids[task_id]})

    for reader, readable in (
      ("session-1", ("a0", "old-child", "old-sibling", "a1", "new-child", "a2", "latest-child")),
      ("nested-actor", ("a0", "old-child", "a1", "new-child", "a2", "latest-child")),
      ("later-actor", ("a1", "new-child", "a2", "latest-child")),
    ):
      for task_id in readable:
        page, error = await _read(reader, task_id)
        assert error is None and page is not None
        assert page["content"] == canonical_json_bytes({"answer": task_id}).decode("utf-8")
        assert page["end"] is True
    for reader, denied in (
      ("later-actor", ("a0", "old-child", "old-sibling", "unrelated")),
      ("nested-actor", ("old-sibling", "unrelated")),
      ("actor-parent", ("a0", "a1", "new-child")),
      ("session-alias", ("a0", "a1", "new-child")),
      ("runner-alias", ("a0", "a1", "new-child")),
      ("other-root", ("a0", "a1", "new-child")),
    ):
      for task_id in denied:
        page, error = await _read(reader, task_id)
        assert page is None and error is not None
        assert error["code"] == "handle_not_granted"

  _run(_case())
