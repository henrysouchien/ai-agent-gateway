from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_workflow_contracts import (
  AdmittedInputBinding,
  AdmittedTask,
  CapabilityBind,
  ContentReadContextView,
  ContentReadGrant,
  ContextViewPolicy,
  InvocationArgumentSelector,
  ToolGrant,
  sha256_digest,
)
from agent_gateway.agent_session_log import (
  AgentSessionLog,
  AgentSessionRef,
  resolve_agent_session_id,
  try_acquire_agent_session_log_write_leases,
)
from agent_gateway.context_builder import SessionContextBuilder
from agent_gateway.execution_snapshot import (
  build_agent_execution_snapshot,
  render_result_instructions,
)
from agent_gateway.research_file_current_projection import (
  RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE,
  ResearchFileCurrentProjectionUnavailable,
  build_current_projection_unavailable_event,
  invalidate_owner_current_session_projections,
  load_research_file_current_projection,
  task_registration_is_current,
)
from agent_gateway.skills import SkillLoader
from agent_gateway.sub_agent import (
  _canonical_result_requirement,
  _ordinary_admitted_task_factory,
  _research_file_id_admitted_input,
  seal_admitted_task_payload,
)


_DOC_ID = "doc:" + "a" * 32
_GENERATION_1 = "11111111-1111-4111-8111-111111111111"
_GENERATION_2 = "22222222-2222-4222-8222-222222222222"


def _run(coro):
  return asyncio.run(coro)


def _canonical_log(base_dir: Path, user_id: str, agent_id: str) -> AgentSessionLog:
  return AgentSessionLog(
    session_ref=AgentSessionRef(
      user_id=user_id,
      agent_id=agent_id,
      agent_session_id=resolve_agent_session_id(user_id, agent_id),
    ),
    base_dir=base_dir,
  )


def _admitted_research_file_registration(
  skills_dir: Path,
  *,
  task_id: str,
  research_file_id: int,
  materialization: str = "inline_exact",
  selector: str = "literal",
) -> dict[str, object]:
  """One real admitted registration for a research file, as the log holds it.

  ``materialization`` picks the context view the workflow provider publishes for
  the same admitted source (`runtime_provider.py` falls back to a content-read
  grant for bytes it will not inline); ``selector`` picks a literal or an
  invocation-argument admitted source over the same admitted content.
  """

  operation = SkillLoader(skills_dir).resolve_operation(None).snapshot
  tool_grant = ToolGrant(
    grant_id="grant:research-file",
    tools=(),
    digest=sha256_digest({"grant_id": "grant:research-file", "tools": []}),
  )
  requirement = _canonical_result_requirement(operation=operation)
  parent_session = SimpleNamespace(
    tenant_id="tenant-a",
    session_id="session-a",
  )
  binding = _research_file_id_admitted_input(
    research_file_id,
    invocation_id=task_id,
    parent_session=parent_session,
  )
  task = _ordinary_admitted_task_factory(
    operation=operation,
    execution_snapshot=build_agent_execution_snapshot(
      operation=operation,
      result_instructions=render_result_instructions(requirement),
      admission_date="2026-09-12",
      max_turns=10,
      timeout_seconds=600,
      client_timeout_seconds=90,
      max_tokens=64_000,
      cost_observation_threshold_usd=50,
      provider_tool_definitions=(),
      max_resume_chain_depth=3,
    ),
    capability_bindings=(),
    tool_grant=tool_grant,
    tool_routes=(),
    model_bind=CapabilityBind(
      schema_version="1.0",
      capability_id="node.explore",
      model_key="codex.gpt-5-6-sol",
      provider="codex",
      upstream_model="gpt-5.6-sol",
      adapter="codex.responses",
      protocol_profile="codex.reasoning",
      route="codex.chatgpt",
      effort="high",
      credential_principal="user",
      credential_ref="user:test:alice:codex",
      run_mode="interactive",
      registry_revision="test-registry.1",
      policy_revision="test-policy.1",
      selection_source="internal_policy",
    ),
    result_requirement=requirement,
    objective="Read the admitted research file.",
    parent_session=parent_session,
    inputs=(binding,),
  )(SimpleNamespace(task_id=task_id))

  payload = task.model_dump(mode="json")
  payload.pop("admitted_task_digest")
  if materialization == "content_read" or selector != "literal":
    read_grant = ContentReadGrant(
      grant_id=f"content-read:research-file-{research_file_id}",
      content_id=binding.source.content.content_id,
      scope="this_task",
      principal_id=task.admitted_task_id,
    )
    request_update: dict[str, object] = {
      "context_policy": ContextViewPolicy(
        preferred="content_read",
        max_bytes=8_000,
        on_overflow="content_read",
      ),
    }
    source_update: dict[str, object] = {"read_grant": read_grant}
    if selector != "literal":
      request_update["selector"] = InvocationArgumentSelector(
        argument_name="research_file_id",
      )
      source_update["source_kind"] = "invocation_argument"
    source_update["request"] = binding.source.request.model_copy(
      update=request_update,
    )
    payload["inputs"] = [AdmittedInputBinding(
      name=binding.name,
      source=binding.source.model_copy(update=source_update),
      context=ContentReadContextView(
        source=binding.context.source,
        read_grant=read_grant,
      ),
    ).model_dump(mode="json")]
    payload["content_read_grants"] = [read_grant.model_dump(mode="json")]
  admitted = AdmittedTask.model_validate(seal_admitted_task_payload(payload))
  return {
    "type": "task_registered",
    "task_id": task_id,
    "metadata": {"admitted_task": admitted.model_dump(mode="json")},
  }


def test_owner_cutoff_preserves_archive_foreign_owner_and_unrelated_retry_work(
  tmp_path: Path,
) -> None:
  owner_log = _canonical_log(tmp_path, "owner-a", "analyst")
  foreign_log = _canonical_log(tmp_path, "owner-b", "analyst")
  _run(owner_log.append({"type": "assistant_message", "text": "owner secret"}))
  _run(foreign_log.append({"type": "assistant_message", "text": "foreign secret"}))
  _run(invalidate_owner_current_session_projections(
    tmp_path,
    owner_user_id="owner-a",
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  ))
  raw_owner, _ = _run(owner_log.query(order="asc"))
  assert any("owner secret" in str(entry.event) for entry in raw_owner)
  projection = _run(load_research_file_current_projection(owner_log))
  current_owner, _ = _run(owner_log.query_current_strict(
    order="asc",
    exclude_entry=projection.excludes,
  ))
  assert current_owner == []
  foreign_markers, _ = _run(foreign_log.query(
    event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
  ))
  assert foreign_markers == []

  future = _run(owner_log.append({"type": "user_message", "content": "new intent"}))
  _run(invalidate_owner_current_session_projections(
    tmp_path,
    owner_user_id="owner-a",
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  ))
  same_generation_markers, _ = _run(owner_log.query(
    event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
  ))
  assert len(same_generation_markers) == 1
  projection = _run(load_research_file_current_projection(owner_log))
  assert not projection.excludes(future)

  _run(owner_log.append({
    "type": "attach",
    "runner_id": "runner-rf-41",
    "context_research_file_id": 41,
  }))
  _run(invalidate_owner_current_session_projections(
    tmp_path,
    owner_user_id="owner-a",
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  ))
  advanced_markers, _ = _run(owner_log.query(
    event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
  ))
  assert len(advanced_markers) == 2

  post_retry = _run(owner_log.append({"type": "user_message", "content": "after retry"}))
  _run(invalidate_owner_current_session_projections(
    tmp_path,
    owner_user_id="owner-a",
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_2,
  ))
  projection = _run(load_research_file_current_projection(owner_log))
  assert projection.excludes(post_retry)


def test_context_builder_replays_only_future_events_after_cutoff(
  tmp_path: Path,
) -> None:
  log = _canonical_log(tmp_path, "owner-a", "analyst")
  _run(log.append({"type": "assistant_message", "text": "deleted secret"}))
  _run(invalidate_owner_current_session_projections(
    tmp_path,
    owner_user_id="owner-a",
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  ))
  _run(log.append({"type": "user_message", "content": "future prompt"}))

  messages = _run(SessionContextBuilder(
    agent_session_log=log,
    tail_window_seconds=None,
  ).build())

  assert messages == [{"role": "user", "content": "future prompt"}]
  assert "deleted secret" not in str(messages)


@pytest.mark.parametrize(
  "mutation",
  [
    {"reason": "wrong"},
    {"current_projection_event_version": 999},
    {"document_id": "doc:not-canonical"},
    {"invalidated_through_seq": "0"},
  ],
)
def test_malformed_cutoff_makes_current_projection_unavailable(
  tmp_path: Path,
  mutation: dict[str, object],
) -> None:
  log = _canonical_log(tmp_path, "owner-a", "analyst")
  event = build_current_projection_unavailable_event(
    invalidated_through_seq=0,
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  )
  event.update(mutation)
  _run(log.append(event))

  with pytest.raises(ResearchFileCurrentProjectionUnavailable):
    _run(load_research_file_current_projection(log))


def test_marker_with_additive_field_still_cuts_off(tmp_path: Path) -> None:
  log = _canonical_log(tmp_path, "owner-a", "analyst")
  _run(log.append({"type": "assistant_message", "text": "deleted secret"}))
  event = build_current_projection_unavailable_event(
    invalidated_through_seq=1,
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  )
  event["future_envelope_field"] = True
  _run(log.append(event))

  projection = _run(load_research_file_current_projection(log))

  assert projection.cutoff_seq == 1
  assert len(projection.marker_seqs) == 1


def test_task_current_projection_rejects_old_registration_and_accepts_future(
  tmp_path: Path,
) -> None:
  log = _canonical_log(tmp_path, "owner-a", "analyst")
  _run(log.append({"type": "task_registered", "task_id": "bg-old"}))
  _run(invalidate_owner_current_session_projections(
    tmp_path,
    owner_user_id="owner-a",
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  ))
  assert not _run(task_registration_is_current(log, "bg-old"))
  _run(log.append({"type": "task_registered", "task_id": "bg-new"}))
  assert _run(task_registration_is_current(log, "bg-new"))


def test_active_writer_lease_blocks_before_cutoff_append(tmp_path: Path) -> None:
  log = _canonical_log(tmp_path, "owner-a", "analyst")
  _run(log.append({"type": "assistant_message", "text": "still current"}))
  lease = try_acquire_agent_session_log_write_leases((log.path,))
  assert lease is not None
  try:
    with pytest.raises(ResearchFileCurrentProjectionUnavailable):
      _run(invalidate_owner_current_session_projections(
        tmp_path,
        owner_user_id="owner-a",
        research_file_ids=(41,),
        document_id=_DOC_ID,
        document_generation=_GENERATION_1,
      ))
  finally:
    lease.release()
  markers, _ = _run(log.query(
    event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
  ))
  assert markers == []


def test_authenticated_storage_alias_selects_captured_log_only(
  tmp_path: Path,
) -> None:
  captured = _canonical_log(tmp_path, "login-user", "research_producer")
  foreign = _canonical_log(tmp_path, "foreign-user", "research_producer")
  _run(captured.append({"type": "assistant_message", "text": "captured secret"}))
  _run(foreign.append({"type": "assistant_message", "text": "foreign text"}))

  _run(invalidate_owner_current_session_projections(
    tmp_path,
    owner_user_id="42",
    owner_user_id_aliases=("login-user",),
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  ))

  captured_projection = _run(load_research_file_current_projection(captured))
  captured_current, _ = _run(captured.query_current_strict(
    exclude_entry=captured_projection.excludes,
  ))
  assert captured_current == []
  foreign_markers, _ = _run(foreign.query(
    event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
  ))
  assert foreign_markers == []


def _erase_research_file_41_document(base_dir: Path) -> None:
  _run(invalidate_owner_current_session_projections(
    base_dir,
    owner_user_id="owner-a",
    research_file_ids=(41,),
    document_id=_DOC_ID,
    document_generation=_GENERATION_1,
  ))


@pytest.mark.parametrize(
  ("materialization", "selector"),
  [
    ("inline_exact", "literal"),
    ("content_read", "literal"),
    ("content_read", "invocation_argument"),
  ],
)
def test_same_generation_retry_invalidates_registration_by_admitted_content(
  tmp_path: Path,
  tmp_path_factory: pytest.TempPathFactory,
  materialization: str,
  selector: str,
) -> None:
  log = _canonical_log(tmp_path, "owner-a", "analyst")
  _run(log.append({"type": "assistant_message", "content": "old private evidence"}))
  _erase_research_file_41_document(tmp_path)
  registration = _run(log.append(_admitted_research_file_registration(
    tmp_path_factory.mktemp("skills"),
    task_id="bg-research-file",
    research_file_id=41,
    materialization=materialization,
    selector=selector,
  )))
  derived = _run(log.append({
    "type": "assistant_message",
    "content": "derived private evidence from RF 41",
  }))

  _erase_research_file_41_document(tmp_path)

  projection = _run(load_research_file_current_projection(log))
  assert len(projection.marker_seqs) == 2
  assert projection.excludes(registration)
  assert projection.excludes(derived)
  assert not _run(task_registration_is_current(log, "bg-research-file"))
  messages = _run(SessionContextBuilder(
    agent_session_log=log,
    tail_window_seconds=None,
  ).build())
  assert messages == []


def test_same_generation_retry_keeps_unrelated_content_read_registration(
  tmp_path: Path,
  tmp_path_factory: pytest.TempPathFactory,
) -> None:
  log = _canonical_log(tmp_path, "owner-a", "analyst")
  _run(log.append({"type": "assistant_message", "content": "old private evidence"}))
  _erase_research_file_41_document(tmp_path)
  registration = _run(log.append(_admitted_research_file_registration(
    tmp_path_factory.mktemp("skills"),
    task_id="bg-research-file",
    research_file_id=42,
    materialization="content_read",
  )))
  derived = _run(log.append({
    "type": "assistant_message",
    "content": "derived evidence from RF 42",
  }))

  _erase_research_file_41_document(tmp_path)

  projection = _run(load_research_file_current_projection(log))
  assert len(projection.marker_seqs) == 1
  assert not projection.excludes(registration)
  assert not projection.excludes(derived)
  assert _run(task_registration_is_current(log, "bg-research-file"))


def test_unresolvable_registration_identity_fails_retry_closed(
  tmp_path: Path,
  tmp_path_factory: pytest.TempPathFactory,
) -> None:
  log = _canonical_log(tmp_path, "owner-a", "analyst")
  _run(log.append({"type": "assistant_message", "content": "old private evidence"}))
  _erase_research_file_41_document(tmp_path)
  event = _admitted_research_file_registration(
    tmp_path_factory.mktemp("skills"),
    task_id="bg-unresolvable",
    research_file_id=41,
    materialization="content_read",
  )
  # Without its admitted content handle the committed identity is unresolvable.
  del event["metadata"]["admitted_task"]["inputs"][0]["source"]["content"]
  _run(log.append(event))

  with pytest.raises(
    ResearchFileCurrentProjectionUnavailable,
    match="malformed admission metadata",
  ):
    _erase_research_file_41_document(tmp_path)

  markers, _ = _run(log.query(
    event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
  ))
  assert len(markers) == 1
