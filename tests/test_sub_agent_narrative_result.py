from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_workflow_contracts import (
  AgentOperationRef,
  AdmittedToolRoute,
  AgentExecutionSnapshot,
  AgentOperationSnapshot,
  AgentResumeMechanics,
  AttemptRef,
  LiveToolCapabilityBinding,
  ContractRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ProjectionInline,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
  ToolGrant,
  ToolGrantEntry,
  ProviderToolDefinition,
  sha256_digest,
)
from model_authority.bind import CapabilityBind

from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.final_narrative_artifact import (
  publish_final_narrative,
  read_final_narrative,
)
from agent_gateway.runner_background_tasks import (
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
)
from agent_gateway.sub_agent import ordinary_admitted_task_factory
from agent_gateway.sub_agent_narrative_result import (
  final_child_visible_text,
  read_task_result_terminal_narrative,
  task_result_from_execution,
)
from agent_gateway.sub_agent_result_evidence import collect_sub_agent_result_evidence
from agent_gateway.task_registry import TaskState, task_state_for_result


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
    role="sub_agent",
    sub_agent_id="child-1",
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
@pytest.mark.parametrize("role", ["writer", "sub_agent"])
async def test_independent_run_narrative_uses_own_durable_response(
  tmp_path, role,
) -> None:
  logical, attempt, provenance = _task_identity()
  log = AgentSessionLog(tmp_path / "independent-run.jsonl")
  sub_agent_id = None if role == "writer" else "child-1"
  runner_id = "independent-run"
  identity = {
    "runner_id": runner_id,
    "role": role,
    "sub_agent_id": sub_agent_id,
  }
  first = await log.append({
    **_assistant_entry(
      "Learning ", seq=1, ordinal=0, terminal=False,
    ).event,
    **identity,
  })
  terminal = await log.append({
    **_assistant_entry(
      "fork narrative.", seq=2, ordinal=1, terminal=True,
      prior_seq=first.seq,
    ).event,
    **identity,
  })
  # Neither another writer run nor a child in the same journal supplies this
  # run's newest terminal segment, even if logical response IDs coincide.
  await log.append({
    **_assistant_entry("Other run.", seq=3, ordinal=0, terminal=True).event,
    "runner_id": "other-run",
    "role": "writer",
  })
  await log.append({
    **_assistant_entry("Nested child.", seq=4, ordinal=0, terminal=True).event,
    "runner_id": runner_id,
    "role": "sub_agent",
    "sub_agent_id": "nested-child",
  })

  visible = await final_child_visible_text(
    log,
    sub_session_id="child-1",
    role=role,
    sub_agent_id=sub_agent_id,
    workspace_dir=str(tmp_path),
    runner_id=runner_id,
  )
  assert visible.text == "Learning fork narrative."
  assert visible.final_narrative is not None
  assert visible.final_narrative.terminal_event_seq == terminal.seq
  assert visible.final_narrative == publish_final_narrative(
    workspace_dir=tmp_path,
    sub_agent_id="child-1",
    terminal_event_seq=terminal.seq,
    text="Learning fork narrative.",
  )
  result = task_result_from_execution(
    (),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=visible.final_narrative,
    stalled=False,
  )
  assert result.execution.status == "succeeded"
  assert read_task_result_terminal_narrative(
    result, workspace_dir=str(tmp_path),
  ) == "Learning fork narrative."


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
    role="sub_agent",
    sub_agent_id="child-1",
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
      role="sub_agent",
      sub_agent_id="child-1",
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
    role="sub_agent",
    sub_agent_id="child-1",
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
    role="sub_agent",
    sub_agent_id="child-1",
    workspace_dir=str(tmp_path),
  )

  result = task_result_from_execution(
    (),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=visible.final_narrative,
    stalled=False,
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
    stalled=False,
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
    role="sub_agent",
    sub_agent_id="child-1",
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
    stalled=False,
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
    role="sub_agent",
    sub_agent_id="child-1",
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
    stalled=False,
  )

  observed = result.evidence.observed_sources
  assert len(observed) == 1
  assert observed[0].kind == "observed_source"
  assert observed[0].excerpt_handle_id is None
  assert all(ref.kind != "citation" for ref in observed)


def _admitted_task(
  *,
  tool_ids: tuple[str, ...] = ("filings_search",),
  resumable: bool = False,
):
  """Build the real admission contract consumed by settlement."""

  logical, _, _ = _task_identity()
  digest = sha256_digest({"fixture": "admission"})
  grant = ToolGrant(
    grant_id="grant-1",
    tools=tuple(
      ToolGrantEntry(tool_id=tool_id, route_id="route-1", effect="read")
      for tool_id in tool_ids
    ),
    digest=sha256_digest({"grant": list(tool_ids)}),
  )
  result_instructions = "Return one exact terminal narrative."
  factory = ordinary_admitted_task_factory(
    operation=AgentOperationSnapshot(
      operation=logical.operation,
      methodology=ContractRef(
        namespace="test",
        name="methodology",
        version="1.0",
        digest=digest,
      ),
      prompt=ContractRef(
        namespace="test",
        name="prompt",
        version="1.0",
        digest=digest,
      ),
      description="Narrative settlement fixture.",
      instructions="Complete the delegated research task.",
      execution_class="test",
      workspace_scope="read_only",
      result_modes=("narrative",),
      resumable=resumable,
    ),
    execution_snapshot=AgentExecutionSnapshot(
      system_prompt=f"2026-09-01\n{result_instructions}",
      admission_date="2026-09-01",
      persisted_methodology_state=None,
      result_instructions=result_instructions,
      client_timeout_seconds=30.0,
      max_tokens=1_024,
      provider_tool_definitions=tuple(
        ProviderToolDefinition(definition={"name": tool_id})
        for tool_id in tool_ids
      ),
      resume_mechanics=AgentResumeMechanics(
        resumable=resumable,
        max_chain_depth=1 if resumable else 0,
        transcript_strategy="durable_reconstruction",
        prompt_strategy="reuse_exact",
        tool_grant_strategy="reissue_exact",
        control_message_strategy="admitted_exact",
      ),
    ),
    capability_bindings=(
      LiveToolCapabilityBinding(
        capability="research-filings.read/v1",
        route_id="route-1",
        tool_ids=tool_ids,
      ),
    ),
    tool_grant=grant,
    tool_routes=tuple(
      AdmittedToolRoute(tool_id=tool_id, origin="local", server_id=None)
      for tool_id in tool_ids
    ),
    model_bind=CapabilityBind(
      schema_version="1.0",
      capability_id="test",
      model_key="test.model",
      provider="anthropic",
      upstream_model="test-model",
      adapter="native",
      protocol_profile="messages.standard",
      route="test",
      effort="none",
      credential_principal="user",
      credential_ref="test",
      run_mode="interactive",
      registry_revision="1.0",
      policy_revision="1.0",
      selection_source="internal_policy",
    ),
    result_requirement=_narrative_requirement(),
    objective="Complete the delegated research task.",
    parent_session=None,
    logical_task_override=logical,
  )
  return factory(SimpleNamespace(task_id="child-1"))


async def _narrative(tmp_path, text: str = "Partial but real answer."):
  visible = await final_child_visible_text(
    _NarrativeLog([
      _assistant_entry(text, seq=11, ordinal=0, terminal=True)
    ]),
    sub_session_id="child-1",
    role="sub_agent",
    sub_agent_id="child-1",
    workspace_dir=str(tmp_path),
  )
  return visible.final_narrative


_STEER_EVENT = {"type": "soft_budget_steer", "kind": "turns", "observed": 3, "threshold": 3}
_FINAL_EVENT = {"type": "soft_budget_final", "kind": "turns", "observed": 5, "threshold": 3}
_STREAM_COMPLETE_EVENT = {
  "type": "stream_complete",
  "usage": {"input_tokens": 40, "output_tokens": 20},
}


@pytest.mark.asyncio
async def test_forced_final_answer_settles_as_an_ordinary_completed_result(
  tmp_path,
) -> None:
  # The soft-budget ladder's steer and forced final are not settlement
  # classes: a child that answered after them settles exactly like one that
  # never met the ladder, and its usage rides the result.
  logical, attempt, provenance = _task_identity()
  narrative = await _narrative(tmp_path)

  def settle(events):
    return task_result_from_execution(
      events,
      logical_task=logical,
      attempt=attempt,
      requirement=_narrative_requirement(),
      provenance=provenance,
      final_narrative=narrative,
      stalled=False,
      admitted_task=_admitted_task(),
    )

  laddered = settle((_STEER_EVENT, _FINAL_EVENT, _STREAM_COMPLETE_EVENT))
  plain = settle((_STREAM_COMPLETE_EVENT,))

  assert laddered.execution.status == "succeeded"
  assert laddered.execution.terminal_reason is None
  assert laddered.values.terminal_narrative is not None
  assert laddered.outcome == plain.outcome
  assert laddered.observation.usage.input_tokens == 40
  assert laddered.observation.usage.output_tokens == 20


_SOFT_BUDGET_ABORT_EVENT = {
  "type": "interrupted",
  "reason": "soft_budget_abort",
  "kind": "turns",
  "observed": 7,
  "threshold": 3,
  "grace_turns": 2,
}


@pytest.mark.asyncio
async def test_resumable_soft_budget_abort_settles_interrupted_turns_exhausted(
  tmp_path,
) -> None:
  # A resumable operation's exhausted ladder is a park, not a failure: the
  # registry reads ``turns_exhausted`` as PARKED.
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (_STEER_EVENT, _FINAL_EVENT, _SOFT_BUDGET_ABORT_EVENT, _STREAM_COMPLETE_EVENT),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path),
    stalled=False,
    admitted_task=_admitted_task(resumable=True),
  )

  assert result.execution.status == "interrupted"
  assert result.execution.terminal_reason == "turns_exhausted"
  assert result.execution.model_dump(mode="json")["next_call"] is None
  assert task_state_for_result(result) == TaskState.PARKED
  assert result.outcome is None
  assert result.values.terminal_narrative is None
  assert result.observation.usage.input_tokens == 40


@pytest.mark.asyncio
async def test_non_resumable_soft_budget_abort_settles_failed_with_its_terminal_reason(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (_STEER_EVENT, _FINAL_EVENT, _SOFT_BUDGET_ABORT_EVENT, _STREAM_COMPLETE_EVENT),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path),
    stalled=False,
    admitted_task=_admitted_task(),
  )

  assert result.execution.status == "failed"
  assert result.execution.terminal_reason == "soft_budget_abort"
  assert result.execution.model_dump(mode="json")["next_call"] is None
  assert task_state_for_result(result) == TaskState.FAILED
  assert result.outcome is None
  assert result.values.terminal_narrative is None
  assert result.observation.usage.input_tokens == 40


@pytest.mark.asyncio
async def test_soft_budget_abort_beside_a_harder_signal_keeps_its_precedence(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    ({"type": "interrupted", "reason": "soft_budget_abort"},),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path),
    stalled=True,
    admitted_task=_admitted_task(),
  )

  assert result.execution.status == "interrupted"
  assert result.execution.terminal_reason is not None
  assert result.execution.terminal_reason.startswith("stalled:")
  assert result.execution.model_dump(mode="json")["next_call"] is None
  assert result.outcome is None
  assert result.values.terminal_narrative is None


@pytest.mark.parametrize("reason", ("stalled", "soft_budget_abort"))
@pytest.mark.parametrize("next_call", (
  None,
  {"tool_name": "resume_agent", "arguments": {"task_id": "child-1"}},
))
def test_terminal_settlement_copies_its_producers_next_call(reason, next_call) -> None:
  logical, attempt, provenance = _task_identity()
  event = (
    {"type": "error", "error_sub_code": "stalled", "error": "No activity"}
    if reason == "stalled"
    else {"type": "interrupted", "reason": reason}
  )
  event["next_call"] = next_call
  result = task_result_from_execution(
    (event,),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=None,
    stalled=reason == "stalled",
    admitted_task=_admitted_task(),
  )

  assert result.execution.terminal_reason is not None
  assert result.execution.terminal_reason.startswith(reason)
  assert result.execution.model_dump(mode="json")["next_call"] == next_call



def test_usage_turns_merge_prior_attempt_without_double_counting_stream_usage() -> None:
  logical, attempt, provenance = _task_identity()
  prior = collect_sub_agent_result_evidence(
    ({"type": "turn_complete"}, {"type": "turn_complete"}),
    durable=False,
  )
  result = task_result_from_execution(
    (
      {"type": "turn_complete", "usage": {"input_tokens": 10}},
      {"type": "turn_complete", "usage": {"input_tokens": 20}},
      {"type": "stream_complete", "usage": {"input_tokens": 30}},
      {"type": "interrupted", "reason": "soft_budget_abort"},
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=None,
    stalled=False,
    prior_evidence=prior,
    admitted_task=_admitted_task(),
  )

  assert result.observation.usage.turns == 4
  assert result.observation.usage.input_tokens == 30


def test_pre_measurement_persisted_task_result_loads_without_rewriting_identity() -> None:
  logical, attempt, provenance = _task_identity()
  result = task_result_from_execution(
    ({"type": "interrupted", "reason": "soft_budget_abort"},),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=None,
    stalled=False,
    admitted_task=_admitted_task(),
  )
  persisted = result.model_dump(mode="json")
  persisted["execution"].pop("next_call", None)
  persisted["observation"]["usage"].pop("turns", None)
  old_digest = sha256_digest({
    key: value for key, value in persisted.items() if key != "task_result_id"
  })
  persisted["task_result_id"] = f"task-result:{old_digest}"

  loaded = TaskResult.model_validate(persisted)

  assert loaded.task_result_id == persisted["task_result_id"]
  assert loaded.execution.next_call is None
  assert loaded.observation.usage.turns == 0
  # A record without the fields reserializes byte-identically.
  reloaded = loaded.model_dump(mode="json")
  assert "next_call" not in reloaded["execution"]
  assert "turns" not in reloaded["observation"]["usage"]
  assert reloaded == persisted


@pytest.mark.asyncio
async def test_unadmitted_evidence_is_a_recorded_gap_not_a_failed_child(
  tmp_path,
) -> None:
  # GC58: usage evidence that cannot be merged is not kept, but the child's
  # own answer still settles; the gap rides the mechanical outcome.
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (
      {"type": "turn_complete", "usage": {"counter": 10**4000}},
      {"type": "turn_complete", "usage": {"counter": 1.0}},
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path),
    stalled=False,
    admitted_task=_admitted_task(),
  )

  assert result.execution.status == "succeeded"
  assert result.values.terminal_narrative is not None
  assert result.outcome is not None
  assert result.outcome.disposition == "partial"
  assert result.outcome.unmet_requirements == ("child_evidence",)

_REFUSAL_EVENT = {
  "type": "refusal",
  "category": "reasoning_extraction",
  "explanation": "The request asks for internal reasoning.",
  "guidance": "Anthropic recommends starting a fresh conversation before continuing.",
}


@pytest.mark.asyncio
async def test_refused_child_settles_interrupted_and_keeps_delivered_narrative(
  tmp_path,
) -> None:
  # A provider refusal is its own disposition: the text the model delivered
  # before the stop is the child's narrative, never a runtime_error.
  logical, attempt, provenance = _task_identity()
  refused = _assistant_entry("Delivered before the stop.", seq=11, ordinal=0, terminal=True)
  refused.event["stop_reason"] = "refusal"
  visible = await final_child_visible_text(
    _NarrativeLog([refused]),
    sub_session_id="child-1",
    role="sub_agent",
    sub_agent_id="child-1",
    workspace_dir=str(tmp_path),
  )
  assert visible.text == "Delivered before the stop."

  result = task_result_from_execution(
    (dict(_REFUSAL_EVENT),),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=visible.final_narrative,
    stalled=False,
    admitted_task=_admitted_task(),
  )

  assert result.execution.status == "interrupted"
  assert result.execution.terminal_reason == "refusal: reasoning_extraction"
  assert result.values.terminal_narrative is not None
  assert read_task_result_terminal_narrative(
    result,
    workspace_dir=str(tmp_path),
  ) == "Delivered before the stop."


@pytest.mark.asyncio
async def test_refusal_beside_a_runtime_error_keeps_failing(tmp_path) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (dict(_REFUSAL_EVENT), {"type": "error", "error": "channel closed"}),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path),
    stalled=False,
    admitted_task=_admitted_task(),
  )

  assert result.execution.status == "failed"
  assert result.execution.terminal_reason == "runtime_error: channel closed"
  assert result.values.terminal_narrative is None


@pytest.mark.asyncio
async def test_clean_execution_derives_a_complete_mechanical_outcome(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (
      {"type": "tool_call_start", "tool_name": "filings_search"},
      {
        "type": "tool_call_complete",
        "tool_name": "filings_search",
        "dispatch": {"outcome": "ok"},
      },
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path, "Complete answer."),
    stalled=False,
    admitted_task=_admitted_task(),
  )

  assert result.execution.status == "succeeded"
  assert result.outcome is not None
  assert result.outcome.disposition == "complete"
  assert result.outcome.assessment_source == "mechanically_derived"


@pytest.mark.asyncio
async def test_failed_source_retrievals_derive_insufficient_evidence(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (
      {"type": "tool_call_start", "tool_name": "filings_search"},
      {
        "type": "tool_call_complete",
        "tool_name": "filings_search",
        "dispatch": {"outcome": "error_rate_limited"},
      },
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path, "I could not read the filings."),
    stalled=False,
    admitted_task=_admitted_task(),
  )

  assert result.outcome is not None
  assert result.outcome.disposition == "insufficient_evidence"


@pytest.mark.asyncio
async def test_settlement_without_an_admitted_task_derives_no_outcome(
  tmp_path,
) -> None:
  # task_entry is Optional at all three settlement sites. No admission in
  # scope means no assessment occurred — never a false ``complete``.
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path, "Unadmitted answer."),
    stalled=False,
  )

  assert result.execution.status == "succeeded"
  assert result.outcome is None


def _fms_complete(tool_name: str, **result_fields: object) -> dict[str, object]:
  payload = {
    "subcommand": tool_name.removeprefix("fms_"),
    "mutation_mode": "model_writer",
    **result_fields,
  }
  return {
    "type": "tool_call_complete",
    "tool_name": tool_name,
    "result": payload,
  }


@pytest.mark.asyncio
async def test_terminal_tool_success_projects_exact_result_without_narrative(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()
  tool_name = "fms_propose_position_initiation_predecision"
  tool_result = {
    "status": "staged",
    "proposal_id": "proposal-1",
    "artifact_ref": "artifacts/PCTY/predecision.json",
  }

  result = task_result_from_execution(
    (
      {"type": "tool_call_start", "tool_name": tool_name},
      {
        "type": "tool_call_complete",
        "tool_name": tool_name,
        "result": tool_result,
        "dispatch": {"outcome": "ok"},
        "is_error": False,
        "error": None,
        "semantic_error": None,
      },
      {"type": "interrupted", "reason": "soft_budget_abort"},
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=None,
    stalled=False,
    admitted_task=_admitted_task(tool_ids=(tool_name,)),
  )

  assert result.execution.status == "succeeded"
  assert result.values.terminal_narrative is None
  assert result.values.projection is not None
  assert result.values.projection.inline_view == {
    "tool_name": tool_name,
    "result": tool_result,
  }


@pytest.mark.asyncio
async def test_named_operation_without_terminal_door_settles_on_its_narrative(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (
      {"type": "tool_call_start", "tool_name": "prepare_model_build"},
      {
        "type": "tool_call_complete",
        "tool_name": "prepare_model_build",
        "dispatch": {"outcome": "ok"},
      },
      {"type": "tool_call_start", "tool_name": "list_research_files"},
      {
        "type": "tool_call_complete",
        "tool_name": "list_research_files",
        "dispatch": {"outcome": "ok"},
      },
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path, "Prepared; the build is next."),
    stalled=False,
    admitted_task=_admitted_task(tool_ids=(
      "prepare_model_build",
      "list_research_files",
      "fms_report_build_model",
    )),
  )

  assert result.execution.status == "succeeded"
  assert result.execution.terminal_reason is None
  assert result.values.terminal_narrative is not None
  assert result.values.projection is None
  assert result.evidence.tools_used == (
    "prepare_model_build",
    "list_research_files",
  )
  assert result.outcome is not None
  assert result.outcome.disposition != "blocked"


@pytest.mark.asyncio
async def test_terminal_door_stop_is_the_child_result(tmp_path) -> None:
  logical, attempt, provenance = _task_identity()
  door = "fms_persist_postcompile_valuation"
  door_result = {
    "subcommand": "persist_postcompile_valuation",
    "mutation_mode": "model_writer",
    "status": "error",
    "gate_code": "STOP",
    "error": {
      "type": "INVALID_JUDGMENT",
      "message": "postcompile-valuation persistence requires the active research file",
    },
  }

  result = task_result_from_execution(
    (
      {"type": "tool_call_start", "tool_name": door},
      {"type": "tool_call_complete", "tool_name": door, "result": door_result},
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path, "The persist door refused."),
    stalled=False,
    admitted_task=_admitted_task(tool_ids=(door,)),
  )

  assert result.execution.status == "succeeded"
  assert result.values.projection is not None
  assert result.values.projection.inline_view == {
    "tool_name": door,
    "result": door_result,
  }
  assert result.execution.terminal_reason is None
  # The refusal is the row's outcome: a workflow consumer reading the
  # settlement does not read a refused door as a completed stage.
  assert result.outcome is not None
  assert result.outcome.disposition == "blocked"
  assert result.outcome.unmet_requirements == (door,)


_CONTEXT_DOOR = "get_postcompile_valuation_context"
_CONTEXT_REFUSAL = {
  "status": "error",
  "error_type": "postcompile_model_version_not_ready",
  "error": "the model version has no accepted valuation policy",
  "next_call": {
    "tool_name": "get_valuation_policy_context",
    "arguments": {"research_file_id": 1},
  },
}


def _context_read_event(result: dict[str, object]) -> dict[str, object]:
  return {
    "type": "tool_call_complete",
    "tool_name": _CONTEXT_DOOR,
    "result": result,
    "dispatch": {"outcome": "ok"},
  }


@pytest.mark.asyncio
async def test_context_door_refusal_reaches_the_parent_verbatim(tmp_path) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (
      {"type": "tool_call_start", "tool_name": _CONTEXT_DOOR},
      _context_read_event(_CONTEXT_REFUSAL),
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path, "The context door refused."),
    stalled=False,
    admitted_task=_admitted_task(tool_ids=(
      _CONTEXT_DOOR,
      "fms_persist_postcompile_valuation",
    )),
  )

  assert result.execution.status == "succeeded"
  assert result.execution.terminal_reason is None
  assert result.outcome is not None
  assert result.outcome.disposition == "blocked"
  assert result.outcome.unmet_requirements == (_CONTEXT_DOOR,)
  envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: pytest.fail("narrative read"),
    read_grant_factory=lambda _source: pytest.fail("handle minted"),
  )
  materialization = envelope.parent_materialization
  assert isinstance(materialization, ProjectionInline)
  assert materialization.value == {
    "tool_name": _CONTEXT_DOOR,
    "result": _CONTEXT_REFUSAL,
  }
  assert envelope.settlement_projection.execution_status == "succeeded"


@pytest.mark.asyncio
async def test_context_refusal_superseded_by_an_accepted_read_keeps_the_narrative(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (
      _context_read_event(_CONTEXT_REFUSAL),
      _context_read_event({"context_ref": "pcvcx1:exact"}),
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path, "Read the context."),
    stalled=False,
    admitted_task=_admitted_task(tool_ids=(
      _CONTEXT_DOOR,
      "fms_persist_postcompile_valuation",
    )),
  )

  assert result.execution.status == "succeeded"
  assert result.values.projection is None
  assert result.values.terminal_narrative is not None


@pytest.mark.asyncio
async def test_ordinary_child_without_declared_door_stays_succeeded(
  tmp_path,
) -> None:
  logical, attempt, provenance = _task_identity()

  result = task_result_from_execution(
    (
      {"type": "tool_call_start", "tool_name": "filings_search"},
      {
        "type": "tool_call_complete",
        "tool_name": "filings_search",
        "dispatch": {"outcome": "ok"},
      },
    ),
    logical_task=logical,
    attempt=attempt,
    requirement=_narrative_requirement(),
    provenance=provenance,
    final_narrative=await _narrative(tmp_path, "Ordinary child answer."),
    stalled=False,
    admitted_task=_admitted_task(),
  )

  assert result.execution.status == "succeeded"
  assert result.outcome is not None
  assert result.outcome.disposition == "complete"
