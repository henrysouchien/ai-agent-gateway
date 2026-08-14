from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResultProvenance,
  sha256_digest,
)

from agent_gateway.final_narrative_artifact import read_final_narrative
from agent_gateway.sub_agent_narrative_result import (
  final_child_visible_text,
  read_task_result_terminal_narrative,
  task_result_from_execution,
)


class _NarrativeLog:
  def __init__(self, entries: list[SimpleNamespace]) -> None:
    self.entries = entries
    self.queries: list[dict[str, object]] = []

  async def query(self, **kwargs):
    self.queries.append(dict(kwargs))
    before_seq = kwargs.get("before_seq")
    values = [
      entry
      for entry in self.entries
      if before_seq is None or entry.seq <= before_seq
    ]
    reverse = kwargs.get("order") == "desc"
    values.sort(key=lambda entry: entry.seq, reverse=reverse)
    offset = int(kwargs.get("cursor") or 0)
    limit = int(kwargs.get("limit") or len(values))
    page = values[offset:offset + limit]
    next_offset = offset + len(page)
    cursor = str(next_offset) if next_offset < len(values) else None
    return page, cursor


def _task_identity():
  operation = AgentOperationRef(
    namespace="research",
    name="example",
    version="1.0",
    digest=sha256_digest({"operation": "research.example/1.0"}),
  )
  logical = OrdinaryDelegationTaskRef(
    delegation_id="delegation-1",
    operation=operation,
  )
  attempt = AttemptRef(
    attempt_number=1,
    attempt_id="attempt-1",
    physical_task_id="child-1",
  )
  digest = sha256_digest({"fixture": "provenance"})
  provenance = TaskResultProvenance(
    admitted_task_digest=digest,
    model_bind_digest=digest,
    capability_binding_digest=digest,
    tool_grant_digest=digest,
  )
  return logical, attempt, provenance


def _assistant_entry(
  text: str,
  *,
  seq: int,
  ordinal: int,
  terminal: bool,
  response_id: str = "response-1",
  prior_seq: int | None = None,
) -> SimpleNamespace:
  event: dict[str, object] = {
    "type": "assistant_message",
    "stop_reason": "end_turn" if terminal else "max_tokens",
    "content_blocks": [{"type": "text", "text": text}],
    "logical_response_id": response_id,
    "logical_response_segment_ordinal": ordinal,
  }
  if prior_seq is not None:
    event["continued_from_assistant_message_seq"] = prior_seq
  return SimpleNamespace(event=event, seq=seq)


def _narrative_requirement() -> ResultRequirement:
  return ResultRequirement(
    mode="narrative",
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )


@pytest.mark.asyncio
async def test_terminal_narrative_is_exact_and_never_prefix_clipped(tmp_path) -> None:
  text = "界" * 15_921
  entry = _assistant_entry(text, seq=7, ordinal=0, terminal=True)

  visible = await final_child_visible_text(
    _NarrativeLog([entry]),
    sub_session_id="child-1",
    workspace_dir=str(tmp_path),
  )

  assert visible.text == text
  assert len(visible.text) == 15_921
  assert visible.final_narrative is not None
  assert visible.final_narrative.content_chars == len(text)
  assert visible.final_narrative.content_bytes == len(text.encode("utf-8"))
  assert read_final_narrative(
    workspace_dir=tmp_path,
    reference=visible.final_narrative,
  ) == text


@pytest.mark.asyncio
async def test_terminal_narrative_materializes_all_lineaged_pages(tmp_path) -> None:
  entries = []
  expected: list[str] = []
  for ordinal in range(70):
    text = f"segment-{ordinal:02d}|"
    expected.append(text)
    entries.append(_assistant_entry(
      text,
      seq=ordinal + 1,
      ordinal=ordinal,
      terminal=ordinal == 69,
      prior_seq=ordinal if ordinal else None,
    ))
  log = _NarrativeLog(entries)

  visible = await final_child_visible_text(
    log,
    sub_session_id="child-1",
    workspace_dir=str(tmp_path),
  )

  assert visible.text == "".join(expected)
  assert len(log.queries) >= 2


@pytest.mark.asyncio
async def test_terminal_narrative_requires_durable_lineage(tmp_path) -> None:
  entry = SimpleNamespace(
    seq=3,
    event={
      "type": "assistant_message",
      "stop_reason": "end_turn",
      "content_blocks": [{"type": "text", "text": "legacy prose"}],
    },
  )

  with pytest.raises(RuntimeError, match="logical-response lineage"):
    await final_child_visible_text(
      _NarrativeLog([entry]),
      sub_session_id="child-1",
      workspace_dir=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_newest_nonterminal_message_never_falls_back_to_stale_answer(
  tmp_path,
) -> None:
  stale = _assistant_entry("stale", seq=1, ordinal=0, terminal=True)
  partial = _assistant_entry("partial", seq=2, ordinal=0, terminal=False)

  visible = await final_child_visible_text(
    _NarrativeLog([stale, partial]),
    sub_session_id="child-1",
    workspace_dir=str(tmp_path),
  )

  assert visible.text == ""
  assert visible.final_narrative is None


@pytest.mark.asyncio
async def test_narrative_execution_returns_canonical_task_result(tmp_path) -> None:
  logical, attempt, provenance = _task_identity()
  visible = await final_child_visible_text(
    _NarrativeLog([
      _assistant_entry("Exact canonical answer.", seq=7, ordinal=0, terminal=True)
    ]),
    sub_session_id="child-1",
    workspace_dir=str(tmp_path),
  )

  result = task_result_from_execution(
    (),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=visible.final_narrative,
    timed_out=False,
    timeout=None,
  )

  assert result.execution.status == "succeeded"
  assert result.logical_task == logical
  assert result.attempt == attempt
  assert read_task_result_terminal_narrative(
    result,
    workspace_dir=str(tmp_path),
  ) == "Exact canonical answer."


def test_failed_execution_never_publishes_partial_canonical_values() -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    ({"type": "run_error", "error": "provider disconnected"},),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=None,
    timed_out=False,
    timeout=None,
  )

  assert result.execution.status == "failed"
  assert result.execution.terminal_reason == "runtime_error: provider disconnected"
  assert result.values.terminal_narrative is None
  assert result.values.projection is None


@pytest.mark.asyncio
async def test_child_source_observations_populate_task_result_evidence(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()
  visible = await final_child_visible_text(
    _NarrativeLog([
      _assistant_entry("Sourced answer.", seq=9, ordinal=0, terminal=True)
    ]),
    sub_session_id="child-1",
    workspace_dir=str(tmp_path),
  )

  result = task_result_from_execution(
    (
      {"type": "tool_call_start", "tool_name": "filings_search"},
      {
        "type": "tool_call_complete",
        "tool_name": "filings_search",
        "final_tool_result_blocks": [
          {"type": "tool_result", "tool_use_id": "tool_1", "content": "{}"},
          {
            "type": "source_envelope",
            "_event_only": True,
            "schema_version": 1,
            "tool_name": "filings_search",
            "sources_for_call": [
              {
                "index": 1,
                "document_id": "edgar:0000789019-26-000012",
                "source_kind": "filing",
                "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
                "produced_by_tool": "filings_search",
              },
            ],
            "excerpt_handles_for_call": [
              {
                "handle_id": "h_0123456789ab",
                "document_id": "edgar:0000789019-26-000012",
                "source_kind": "filing",
              },
            ],
          },
        ],
      },
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=visible.final_narrative,
    timed_out=False,
    timeout=None,
  )

  assert result.execution.status == "succeeded"
  assert result.evidence.tools_used == ("filings_search",)
  observed = result.evidence.observed_sources
  assert len(observed) == 1
  assert observed[0].kind == "observed_source"
  assert observed[0].source_kind == "filing"
  assert observed[0].document_id == "edgar:0000789019-26-000012"
  assert observed[0].produced_by_tool == "filings_search"
  assert observed[0].excerpt_handle_id == "h_0123456789ab"


@pytest.mark.asyncio
async def test_no_citation_context_child_yields_no_fabricated_citations(
  tmp_path,
) -> None:
  """Suppressed-context observations stay observations: no citation refs."""

  logical, attempt, provenance = _task_identity()
  visible = await final_child_visible_text(
    _NarrativeLog([
      _assistant_entry("Observed answer.", seq=11, ordinal=0, terminal=True)
    ]),
    sub_session_id="child-1",
    workspace_dir=str(tmp_path),
  )

  result = task_result_from_execution(
    (
      {
        "type": "tool_call_complete",
        "tool_name": "filings_search",
        "final_tool_result_blocks": [
          {"type": "tool_result", "tool_use_id": "tool_1", "content": "{}"},
          {
            "type": "source_observation",
            "_event_only": True,
            "schema_version": 1,
            "tool_name": "filings_search",
            "observed_sources": [
              {
                "document_id": "edgar:0000789019-26-000012",
                "source_kind": "filing",
              },
            ],
          },
        ],
      },
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=visible.final_narrative,
    timed_out=False,
    timeout=None,
  )

  observed = result.evidence.observed_sources
  assert len(observed) == 1
  assert observed[0].kind == "observed_source"
  assert observed[0].excerpt_handle_id is None
  assert all(ref.kind != "citation" for ref in observed)
