"""Resolve declared workflow refs from granted durable handles.

The readback bag is the one minter. A consumer does not receive a pasted copy
of that bag: it is granted the producer's content handle, this owner admits
the read, and the prompt lines are rendered from that read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agent_workflow_contracts import AgentCompletionEnvelope, PipelineStageTaskRef

from .admission import HandleOwners, HandleRequest, ParentContext, Refusal, admit
from .agent_result_content import read_admitted_result_content
from .task_registry import job_output_handles

_HEADER = (
  "Workflow-injected refs from prior tool results; pass each one unchanged "
  "where this skill names it:"
)


def render_resolved_ref_lines(
  refs: Mapping[str, Any],
  declared_keys: Sequence[str],
) -> tuple[str, ...]:
  """The prompt lines for declared keys this read actually returned."""

  bound = [key for key in declared_keys if key in refs]
  if not bound:
    return ()
  return (
    _HEADER,
    *(
      f"{key}={json.dumps(refs[key], sort_keys=True, separators=(',', ':'))}"
      for key in bound
    ),
  )


async def resolve_handle_refs(
  *,
  reader: ParentContext,
  durable_log: Any,
  workspace_dir: str,
  key_handles: Mapping[str, str] | None = None,
  handles: Sequence[str] = (),
  declared_keys: Sequence[str] = (),
) -> dict[str, Any] | Refusal:
  """Read granted producer bags and return the declared keys they carry.

  ``key_handles`` names the exact producer handle for each key (the pipeline's
  latest-producer map). Without it, each handle is read and a later bag
  overrides an earlier one for a declared key. An admission refusal is
  returned whole: the caller does not render a partial bag.
  """

  if reader.job_id is None:
    return Refusal(
      "session_unauthenticated",
      "admission tenant",
      "handle reads require an authenticated reader identity",
    )
  if key_handles is not None:
    wanted = dict(key_handles)
    unique = _unique(wanted.values())
  else:
    # A mixed grant bag is not a stage-ref bag. Only a completion whose
    # journal entry names this handle and whose logical task is a pipeline
    # stage is read. Spill, workflow-membership and ordinary completion
    # handles stay grants the child reads itself.
    wanted = None
    unique = await pipeline_stage_completion_handles(durable_log, handles)
  if not unique:
    return {}
  outputs = await job_output_handles(durable_log, job_id=reader.job_id)
  bags: dict[str, dict[str, Any]] = {}
  for handle in unique:
    admitted = admit(
      HandleRequest(handle=handle, reader_job_id=reader.job_id),
      reader,
      owners=HandleOwners(outputs=outputs),
    )
    if isinstance(admitted, Refusal):
      return admitted
    text = await read_admitted_result_content(
      durable_log,
      content_id=handle,
      workspace_dir=workspace_dir,
    )
    bags[handle] = _read_back_bag(text)
  if wanted is not None:
    return {
      key: bags[handle][key]
      for key, handle in wanted.items()
      if key in bags.get(handle, {})
    }
  resolved: dict[str, Any] = {}
  for handle in unique:
    bag = bags.get(handle) or {}
    for key in declared_keys:
      if key in bag:
        resolved[key] = bag[key]
  return resolved


def _unique(handles: Any) -> list[str]:
  seen: list[str] = []
  for handle in handles:
    if isinstance(handle, str) and handle and handle not in seen:
      seen.append(handle)
  return seen


def _completion_content_id(envelope: AgentCompletionEnvelope) -> str | None:
  source = getattr(envelope.parent_materialization, "source", None)
  content_id = getattr(source, "content_id", None)
  return content_id if isinstance(content_id, str) and content_id else None


async def pipeline_stage_completion_handles(
  durable_log: Any,
  handles: Sequence[str],
) -> list[str]:
  """Handles whose naming completion is a pipeline-stage task.

  The decision is the ``agent_completion`` entry whose content id is the
  handle, and that entry's ``task_result_ref.logical_task``. Handle string
  shape is not a classification.
  """

  stage_handles: list[str] = []
  for handle in _unique(handles):
    if await _names_pipeline_stage_completion(durable_log, handle):
      stage_handles.append(handle)
  return stage_handles


async def _names_pipeline_stage_completion(
  durable_log: Any,
  handle: str,
) -> bool:
  entries, _cursor = await durable_log.query(
    event_types={"agent_completion"},
    contains_text=handle,
    order="desc",
  )
  for item in entries:
    event = item.event
    if event.get("type") != "agent_completion":
      continue
    try:
      envelope = AgentCompletionEnvelope.model_validate(event.get("envelope"))
    except Exception:
      continue
    if _completion_content_id(envelope) != handle:
      continue
    if isinstance(envelope.task_result_ref.logical_task, PipelineStageTaskRef):
      return True
  return False


def _read_back_bag(text: str) -> dict[str, Any]:
  try:
    document = json.loads(text)
  except json.JSONDecodeError:
    return {}
  if not isinstance(document, dict):
    return {}
  refs = document.get("read_back_refs")
  if not isinstance(refs, dict):
    return {}
  return {key: value for key, value in refs.items() if isinstance(key, str)}


__all__ = [
  "pipeline_stage_completion_handles",
  "render_resolved_ref_lines",
  "resolve_handle_refs",
]
