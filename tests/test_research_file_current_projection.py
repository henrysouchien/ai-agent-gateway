from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_gateway.agent_session_log import (
  AgentSessionLog,
  AgentSessionRef,
  resolve_agent_session_id,
  try_acquire_agent_session_log_write_leases,
)
from agent_gateway.context_builder import SessionContextBuilder
from agent_gateway.research_file_current_projection import (
  RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE,
  ResearchFileCurrentProjectionUnavailable,
  build_current_projection_unavailable_event,
  invalidate_owner_current_session_projections,
  load_research_file_current_projection,
  task_registration_is_current,
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
    {"unexpected": True},
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
