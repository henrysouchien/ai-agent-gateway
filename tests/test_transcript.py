from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.transcript import (
  child_run_segment_for_task,
  reconstruct_child_run_lineage,
  reconstruct_messages_for_task,
)


def _run(coro: Any) -> Any:
  return asyncio.run(coro)


async def _append_child_run(
  log: AgentSessionLog,
  *,
  task_id: str,
  sub_agent_id: str,
  runner_id: str,
  message: str,
  original_task_id: str | None = None,
  complete: bool,
) -> None:
  registration = {
    "type": "task_registered",
    "task_id": task_id,
    "task_type": "background",
    "sub_agent_id": sub_agent_id,
    "runner_id": "parent-runner",
    "role": "writer",
  }
  if original_task_id is not None:
    registration["original_task_id"] = original_task_id
  await log.append(registration)
  await log.append(
    {
      "type": "attach",
      "sub_agent_id": sub_agent_id,
      "runner_id": runner_id,
      "role": "sub_agent",
    }
  )
  await log.append(
    {
      "type": "user_message",
      "sub_agent_id": sub_agent_id,
      "runner_id": runner_id,
      "role": "sub_agent",
      "content": message,
    }
  )
  if complete:
    await log.append(
      {
        "type": "task_completed",
        "task_id": task_id,
        "sub_agent_id": sub_agent_id,
        "runner_id": "parent-runner",
        "role": "writer",
        "final_state": "completed",
      }
    )


def test_child_run_segment_excludes_reused_sub_agent_id_outside_window(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "reused-sub-agent.jsonl")
    await _append_child_run(
      log,
      task_id="bg_0",
      sub_agent_id="sub0:shared-session",
      runner_id="runner-child-original",
      message="original run",
      complete=False,
    )
    await _append_child_run(
      log,
      task_id="bg_1",
      sub_agent_id="sub0:shared-session",
      runner_id="runner-child-reused",
      message="unrelated reused run",
      complete=True,
    )

    segment = await child_run_segment_for_task(log, "bg_0")
    assert segment is not None
    assert segment.runner_id == "runner-child-original"
    assert {entry.event.get("runner_id") for entry in segment.entries} == {
      "runner-child-original"
    }
    assert await reconstruct_messages_for_task(log, "bg_0") == [
      {"role": "user", "content": "original run"}
    ]

  _run(_case())


def test_child_run_segment_fails_safe_without_child_attach(tmp_path: Path) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "missing-attach.jsonl")
    await log.append(
      {
        "type": "task_registered",
        "task_id": "bg_legacy",
        "sub_agent_id": "sub0:legacy",
      }
    )
    await log.append(
      {
        "type": "user_message",
        "sub_agent_id": "sub0:legacy",
        "content": "must not be inferred",
      }
    )

    segment = await child_run_segment_for_task(log, "bg_legacy")
    assert segment is not None
    assert segment.runner_id is None
    assert segment.entries == ()
    assert await reconstruct_messages_for_task(log, "bg_legacy") == []

  _run(_case())


def test_reconstruct_child_run_lineage_orders_two_hops_root_to_immediate(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "lineage.jsonl")
    await _append_child_run(
      log,
      task_id="bg_root",
      sub_agent_id="sub0:session",
      runner_id="runner-root",
      message="root",
      complete=True,
    )
    await _append_child_run(
      log,
      task_id="bg_root_r1",
      original_task_id="bg_root",
      sub_agent_id="sub0:session",
      runner_id="runner-r1",
      message="first resume",
      complete=True,
    )
    await _append_child_run(
      log,
      task_id="bg_root_r2",
      original_task_id="bg_root_r1",
      sub_agent_id="sub0:session",
      runner_id="runner-r2",
      message="second resume",
      complete=False,
    )

    lineage = await reconstruct_child_run_lineage(log, "bg_root_r2")
    assert [segment.task_id for segment in lineage] == [
      "bg_root",
      "bg_root_r1",
      "bg_root_r2",
    ]
    assert [segment.runner_id for segment in lineage] == [
      "runner-root",
      "runner-r1",
      "runner-r2",
    ]
    assert [
      entry.event["content"]
      for segment in lineage
      for entry in segment.entries
      if entry.event.get("type") == "user_message"
    ] == ["root", "first resume", "second resume"]
    assert await reconstruct_messages_for_task(log, "bg_root_r2") == [
      {"role": "user", "content": "root"},
      {"role": "user", "content": "first resume"},
      {"role": "user", "content": "second resume"},
    ]

  _run(_case())
