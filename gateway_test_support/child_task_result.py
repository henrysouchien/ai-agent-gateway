from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from agent_workflow_contracts import (
  TERMINAL_NARRATIVE_CONTRACT,
  ActivityHandle,
  ContentHandle,
  ExecutionSettlement,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  TranscriptHandle,
  UsageObservation,
)


def succeeded_child_task_result(spawn_kwargs: Mapping[str, Any]) -> TaskResult:
  """The canonical result a stub ``spawn_sub_agent`` returns for a child.

  ``spawn_kwargs`` are the keyword arguments ``run_agent`` passed the runner;
  the result settles that exact admitted attempt as succeeded with the
  terminal narrative ``ok``.
  """

  attempt = spawn_kwargs["attempt"]
  narrative = b"ok"
  narrative_sha256 = hashlib.sha256(narrative).hexdigest()
  return TaskResult(
    task_result_id=f"result:{attempt.attempt_id}",
    logical_task=spawn_kwargs["logical_task"],
    attempt=attempt,
    execution=ExecutionSettlement(status="succeeded"),
    values=TaskResultValues(
      terminal_narrative=ContentHandle(
        content_id=f"sha256:{narrative_sha256}",
        content_sha256=narrative_sha256,
        content_bytes=len(narrative),
        content_chars=len(narrative),
        contract=TERMINAL_NARRATIVE_CONTRACT,
        media_type="text/plain; charset=utf-8",
        encoding="utf-8",
        retention="durable",
      ),
    ),
    observation=TaskObservation(
      transcript=TranscriptHandle(
        kind="child_transcript",
        owner_id=attempt.physical_task_id,
      ),
      activity=ActivityHandle(
        kind="child_activity",
        owner_id=attempt.physical_task_id,
      ),
      usage=UsageObservation(),
    ),
    provenance=TaskResultProvenance.model_validate(
      spawn_kwargs["result_provenance"]
    ),
  )
