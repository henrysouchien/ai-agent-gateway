from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.event_log import EventLog  # noqa: E402
from agent_gateway.runner_background_lifecycle import (  # noqa: E402
  RunnerBackgroundLifecycleMixin,
  StrictBackgroundTaskDrainUnavailable,
)
from agent_gateway.server_chat_helpers import (  # noqa: E402
  ResearchFileSessionDrainUnavailable,
  quiesce_research_file_sessions,
)
from agent_gateway.session import GatewaySession, SessionStream  # noqa: E402
from agent_gateway.task_registry import TaskRegistry, TaskState  # noqa: E402


def _session(
  *,
  session_id: str,
  user_id: str,
  risk_user_id: int,
  research_file_id: int,
  parent_task: asyncio.Task[Any],
  runtime_owner: Any,
) -> GatewaySession:
  session = GatewaySession(
    session_id=session_id,
    api_key_hash="hash",
    created_at=1,
    expires_at=9999999999,
    user_id=user_id,
    risk_user_id=risk_user_id,
  )
  session.stream_active = True
  session.active_turn = SessionStream(
    event_log=EventLog(),
    runner_task=parent_task,
    research_file_id=research_file_id,
    runtime_owner=runtime_owner,
  )
  return session


@pytest.mark.asyncio
async def test_quiesce_matches_storage_owner_and_file_only() -> None:
  cancelled: list[str] = []

  async def parent(name: str) -> None:
    try:
      await asyncio.Future()
    except asyncio.CancelledError:
      cancelled.append(name)
      raise

  owner = SimpleNamespace(
    cancel_and_require_background_tasks_drained=lambda: _async_none()
  )
  target = _session(
    session_id="target",
    user_id="slug-alice",
    risk_user_id=17,
    research_file_id=41,
    parent_task=asyncio.create_task(parent("target")),
    runtime_owner=owner,
  )
  other_file = _session(
    session_id="other-file",
    user_id="slug-alice",
    risk_user_id=17,
    research_file_id=42,
    parent_task=asyncio.create_task(parent("other-file")),
    runtime_owner=owner,
  )
  other_owner = _session(
    session_id="other-owner",
    user_id="slug-bob",
    risk_user_id=18,
    research_file_id=41,
    parent_task=asyncio.create_task(parent("other-owner")),
    runtime_owner=owner,
  )
  try:
    await asyncio.sleep(0)
    await quiesce_research_file_sessions(
      (target, other_file, other_owner),
      owner_user_id="17",
      research_file_ids=(41,),
    )

    assert cancelled == ["target"]
    assert target.active_turn is None
    assert other_file.active_turn is not None
    assert other_owner.active_turn is not None
  finally:
    remaining = [
      session.active_turn.runner_task
      for session in (other_file, other_owner)
      if session.active_turn is not None
    ]
    for task in remaining:
      task.cancel()
    await asyncio.gather(*remaining, return_exceptions=True)


@pytest.mark.asyncio
async def test_quiesce_keeps_active_turn_discoverable_when_strict_drain_fails(
) -> None:
  async def parent() -> None:
    await asyncio.Future()

  class UnavailableOwner:
    async def cancel_and_require_background_tasks_drained(self) -> None:
      raise StrictBackgroundTaskDrainUnavailable("private detail")

  session = _session(
    session_id="target",
    user_id="alice",
    risk_user_id=0,
    research_file_id=41,
    parent_task=asyncio.create_task(parent()),
    runtime_owner=UnavailableOwner(),
  )
  await asyncio.sleep(0)

  with pytest.raises(
    ResearchFileSessionDrainUnavailable,
    match="document_erase_incomplete",
  ):
    await quiesce_research_file_sessions(
      (session,),
      owner_user_id="alice",
      research_file_ids=(41,),
    )

  assert session.active_turn is not None


@pytest.mark.asyncio
async def test_strict_background_drain_rejects_stubborn_child_then_converges(
) -> None:
  release = asyncio.Event()

  async def stubborn_child() -> None:
    while not release.is_set():
      try:
        await release.wait()
      except asyncio.CancelledError:
        continue

  registry = TaskRegistry()
  entry = registry.register("background_agent")
  registry.transition(entry.task_id, TaskState.RUNNING)
  child = asyncio.create_task(stubborn_child())
  entry.asyncio_task = child
  owner = SimpleNamespace(
    _task_registry=registry,
    _background_cancel_drain_timeout_seconds=0.01,
  )
  strict_drain = (
    RunnerBackgroundLifecycleMixin
    .cancel_and_require_background_tasks_drained
  )
  try:
    await asyncio.sleep(0)
    with pytest.raises(StrictBackgroundTaskDrainUnavailable):
      await strict_drain(owner)
    assert child.done() is False

    release.set()
    await asyncio.wait_for(child, timeout=1)
    await strict_drain(owner)
  finally:
    release.set()
    if not child.done():
      child.cancel()
    await asyncio.gather(child, return_exceptions=True)


@pytest.mark.asyncio
async def test_parent_activity_lease_is_held_until_last_child_finishes() -> None:
  release_child = asyncio.Event()
  released_leases: list[str] = []

  async def child_writer() -> None:
    await release_child.wait()

  registry = TaskRegistry()
  entry = registry.register("background_agent")
  registry.transition(entry.task_id, TaskState.RUNNING)
  child = asyncio.create_task(child_writer())
  entry.asyncio_task = child
  owner = SimpleNamespace(_task_registry=registry)
  lease = SimpleNamespace(release=lambda: released_leases.append("released"))

  RunnerBackgroundLifecycleMixin.bind_research_file_activity_lease(
    owner,
    lease,
  )
  RunnerBackgroundLifecycleMixin._release_research_file_activity_after_children(
    owner
  )
  assert released_leases == []

  release_child.set()
  await child
  await asyncio.sleep(0)
  assert released_leases == ["released"]


@pytest.mark.asyncio
async def test_selected_content_activity_is_held_until_last_child_finishes(
) -> None:
  release_child = asyncio.Event()
  released_leases: list[str] = []

  async def child_writer() -> None:
    await release_child.wait()

  registry = TaskRegistry()
  entry = registry.register("background_agent")
  registry.transition(entry.task_id, TaskState.RUNNING)
  child = asyncio.create_task(child_writer())
  entry.asyncio_task = child
  owner = SimpleNamespace(_task_registry=registry)
  lease = SimpleNamespace(release=lambda: released_leases.append("released"))

  RunnerBackgroundLifecycleMixin.bind_selected_content_activity_lease(
    owner,
    lease,
  )
  RunnerBackgroundLifecycleMixin._release_selected_content_activity_after_children(
    owner
  )
  assert released_leases == []

  release_child.set()
  await child
  await asyncio.sleep(0)
  assert released_leases == ["released"]


async def _async_none() -> None:
  return None
