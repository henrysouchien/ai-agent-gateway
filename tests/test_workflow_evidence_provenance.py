from __future__ import annotations

import hashlib

from agent_workflow_contracts import (
  ActivityHandle,
  AgentOperationRef,
  AttemptRef,
  ContentHandle,
  ContractRef,
  EvidenceObservation,
  ExecutionSettlement,
  ObservedSourceEvidenceRef,
  OrdinaryDelegationTaskRef,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  TranscriptHandle,
  sha256_digest,
)

from agent_gateway.workflow_evidence_provenance import (
  WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY,
  build_workflow_evidence_projection,
  guard_visible_tools_used,
  register_workflow_evidence_projection,
)


def _handle(text: str) -> ContentHandle:
  raw = text.encode("utf-8")
  sha = hashlib.sha256(raw).hexdigest()
  return ContentHandle(
    content_id=f"sha256:{sha}",
    content_sha256=sha,
    content_bytes=len(raw),
    content_chars=len(text),
    contract=ContractRef(
      namespace="test",
      name="report",
      version="1.0",
      digest=sha256_digest("report-contract"),
    ),
    media_type="text/plain; charset=utf-8",
    encoding="utf-8",
    retention="durable",
  )


def _settled_result(
  *,
  status: str = "succeeded",
  tools_used: tuple[str, ...] = (),
  observed_sources: tuple[ObservedSourceEvidenceRef, ...] = (),
) -> TaskResult:
  digest = sha256_digest({"fixture": "provenance"})
  execution = (
    ExecutionSettlement(status="succeeded")
    if status == "succeeded"
    else ExecutionSettlement(status="failed", terminal_reason="runtime_error")
  )
  return TaskResult(
    task_result_id="task-result-1",
    logical_task=OrdinaryDelegationTaskRef(
      delegation_id="delegation-1",
      operation=AgentOperationRef(
        namespace="research",
        name="example",
        version="1.0",
        digest=sha256_digest({"operation": "research.example/1.0"}),
      ),
    ),
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id="attempt-1",
      physical_task_id="child-1",
    ),
    execution=execution,
    evidence=EvidenceObservation(
      observed_sources=observed_sources,
      tools_used=tools_used,
    ),
    values=(
      TaskResultValues(terminal_narrative=_handle("Exact answer."))
      if status == "succeeded"
      else TaskResultValues()
    ),
    observation=TaskObservation(
      transcript=TranscriptHandle(kind="child_transcript", owner_id="child-1"),
      activity=ActivityHandle(kind="child_activity", owner_id="child-1"),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=digest,
      model_bind_digest=digest,
      capability_binding_digest=digest,
      tool_grant_digest=digest,
    ),
  )


def _observed() -> ObservedSourceEvidenceRef:
  return ObservedSourceEvidenceRef(
    source_kind="filing",
    document_id="edgar:0000789019-26-000012",
    produced_by_tool="filings_search",
    source_url="https://www.sec.gov/Archives/msft-10k.htm",
  )


def test_projection_collects_succeeded_node_evidence_only() -> None:
  payload = build_workflow_evidence_projection(
    "workflow-1",
    (
      _settled_result(
        tools_used=("filings_search", "code_execute"),
        observed_sources=(_observed(),),
      ),
      _settled_result(
        status="failed",
        tools_used=("web_fetch",),
      ),
    ),
  )

  assert payload == {
    "workflow_run_id": "workflow-1",
    "evidence_tools": ["filings_search", "code_execute"],
    "observed_sources": [
      {
        "source_kind": "filing",
        "document_id": "edgar:0000789019-26-000012",
        "produced_by_tool": "filings_search",
        "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
      },
    ],
  }


def test_projection_is_none_without_evidence() -> None:
  assert build_workflow_evidence_projection("workflow-1", ()) is None
  assert build_workflow_evidence_projection(
    "workflow-1",
    (_settled_result(),),
  ) is None
  assert build_workflow_evidence_projection("", (_settled_result(),)) is None


def test_registration_validates_and_keys_by_run_identity() -> None:
  store: dict[str, dict[str, object]] = {}
  register_workflow_evidence_projection(store, None)
  register_workflow_evidence_projection(store, {"workflow_run_id": ""})
  register_workflow_evidence_projection(
    store,
    {"workflow_run_id": "workflow-1", "evidence_tools": "filings_search"},
  )
  assert store == {}

  register_workflow_evidence_projection(
    store,
    {
      "workflow_run_id": "workflow-1",
      "evidence_tools": ["filings_search", "filings_search", ""],
      "observed_sources": [
        {"source_kind": "filing", "document_id": "edgar:1"},
        {"source_kind": "", "document_id": "edgar:2"},
        "not-a-record",
      ],
    },
  )

  assert store["workflow-1"]["evidence_tools"] == ["filings_search"]
  assert store["workflow-1"]["observed_sources"] == [
    {"source_kind": "filing", "document_id": "edgar:1"},
  ]


def test_guard_receives_workflow_evidence_projection() -> None:
  store: dict[str, dict[str, object]] = {}
  register_workflow_evidence_projection(
    store,
    {
      "workflow_run_id": "workflow-1",
      "evidence_tools": ["filings_search", "get_financials", "code_execute"],
      "observed_sources": [
        {"source_kind": "filing", "document_id": "edgar:1"},
      ],
    },
  )

  merged = guard_visible_tools_used(["workflow_run"], store.values())

  # Child retrieval provenance is visible; child arithmetic verification is
  # not transferable to the parent's final arithmetic.
  assert merged == ["workflow_run", "filings_search", "get_financials"]
  assert "code_execute" not in merged


def test_guard_view_is_unchanged_without_registered_provenance() -> None:
  assert guard_visible_tools_used(["filings_read"], ()) == ["filings_read"]


def test_private_result_key_is_stable() -> None:
  assert WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY == "_workflow_evidence_projection"
