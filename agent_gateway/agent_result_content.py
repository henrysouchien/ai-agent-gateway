from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from typing import Any

from agent_workflow_contracts import (
  AgentCompletionEnvelope,
  AuthoredSummaryWithResultHandle,
  ContentHandle,
  ContentReadGrant,
  ResultHandle,
  TaskResult,
  TaskResultRef,
  canonical_json_bytes,
)

from .final_narrative_artifact import (
  FinalNarrativeArtifactError,
  read_final_narrative_by_content_handle,
)
from .runner_background_tasks import WORKFLOW_TASK_METADATA_KEY
from .text_paging import json_bounded_text_prefix


AGENT_RESULT_CONTENT_CHUNK_MAX_BYTES = 24_000
AGENT_RESULT_CONTENT_PAGE_MAX_BYTES = 32_000
AGENT_RESULT_CONTENT_MAX_SEQUENCE = 9_007_199_254_740_991
_CONTENT_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class _AuthorizedResultContent:
  handle: ContentHandle
  task_result: TaskResult
  value_kind: Literal["terminal_narrative", "projection"]


def make_get_agent_result_content_tool_def() -> dict[str, Any]:
  return {
    "name": "get_agent_result_content",
    "description": (
      "Read exact canonical result content through the content handle and "
      "direct-parent read grant returned by run_agent. This supports terminal "
      "narrative text and canonical projection JSON. Continue with "
      "next_cursor.after_char until end=true."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "content_id": {
          "type": "string",
          "pattern": r"^sha256:[0-9a-f]{64}$",
          "description": "Exact parent_materialization.source.content_id.",
        },
        "read_grant_id": {
          "type": "string",
          "minLength": 1,
          "maxLength": 512,
          "description": "Exact parent_materialization.read_grant.grant_id.",
        },
        "after_char": {
          "type": "integer",
          "minimum": 0,
          "maximum": AGENT_RESULT_CONTENT_MAX_SEQUENCE,
          "description": (
            "Character cursor from the prior page's next_cursor.after_char."
          ),
        },
      },
      "required": ["content_id", "read_grant_id"],
      "additionalProperties": False,
    },
  }


def make_get_agent_result_content_handler(runner_ref: list[Any]):
  async def _handle_get_agent_result_content(
    tool_input: dict[str, Any],
    **_kwargs: Any,
  ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    runner = runner_ref[0]
    if runner is None:
      return None, {
        "code": "internal_error",
        "message": "Agent runner not initialized",
      }
    parsed, error = _parse_request(tool_input)
    if error is not None:
      return None, error
    assert parsed is not None
    content_id, read_grant_id, after_char = parsed
    durable_log = getattr(runner, "_agent_session_log", None)
    workspace_dir = getattr(runner, "_workspace_dir", None)
    gateway_session_id = str(
      getattr(runner, "_gateway_session_id", "") or ""
    )
    principal_id = str(getattr(runner, "_runner_id", "") or "")
    if (
      durable_log is None
      or workspace_dir is None
      or not gateway_session_id
      or not principal_id
    ):
      return None, {
        "code": "result_content_unavailable",
        "message": "Durable agent result content storage is unavailable",
      }
    authorized, authorization_error = await _authorized_result_content(
      durable_log,
      gateway_session_id=gateway_session_id,
      principal_id=principal_id,
      content_id=content_id,
      read_grant_id=read_grant_id,
    )
    if authorization_error is not None:
      return None, authorization_error
    assert authorized is not None
    try:
      text = _materialize_exact_content(
        authorized,
        workspace_dir=workspace_dir,
      )
      page = _page_result_content(
        authorized=authorized,
        text=text,
        after_char=after_char,
      )
    except (FinalNarrativeArtifactError, ValueError):
      return None, {
        "code": "result_content_unavailable",
        "message": "Agent result content is unavailable or failed verification",
      }
    return page, None

  return _handle_get_agent_result_content


def _parse_request(
  tool_input: Any,
) -> tuple[tuple[str, str, int] | None, dict[str, Any] | None]:
  if type(tool_input) is not dict:
    return None, {
      "code": "invalid_input",
      "message": "get_agent_result_content input must be an exact object",
    }
  unexpected = sorted(
    set(tool_input) - {"content_id", "read_grant_id", "after_char"}
  )
  if unexpected:
    return None, {
      "code": "invalid_input",
      "message": "Unexpected fields: " + ", ".join(unexpected),
    }
  content_id = tool_input.get("content_id")
  if (
    type(content_id) is not str
    or _CONTENT_ID_RE.fullmatch(content_id) is None
  ):
    return None, {
      "code": "invalid_input",
      "message": "content_id must be an exact sha256 content identity",
    }
  read_grant_id = tool_input.get("read_grant_id")
  if (
    type(read_grant_id) is not str
    or not read_grant_id.strip()
    or len(read_grant_id) > 512
  ):
    return None, {
      "code": "invalid_input",
      "message": "read_grant_id must be a non-empty direct-parent grant ID",
    }
  after_char = tool_input.get("after_char", 0)
  if (
    type(after_char) is not int
    or after_char < 0
    or after_char > AGENT_RESULT_CONTENT_MAX_SEQUENCE
  ):
    return None, {
      "code": "invalid_input",
      "message": "after_char must be a non-negative integer",
    }
  return (content_id, read_grant_id.strip(), after_char), None


async def _authorized_result_content(
  durable_log: Any,
  *,
  gateway_session_id: str,
  principal_id: str,
  content_id: str,
  read_grant_id: str,
) -> tuple[_AuthorizedResultContent | None, dict[str, Any] | None]:
  candidates: list[_AuthorizedResultContent] = []
  completion_entries, _ = await durable_log.query(
    event_types={"agent_completion"},
    contains_text=content_id,
    order="desc",
  )
  for completion_entry in completion_entries:
    completion = completion_entry.event
    if completion.get("type") != "agent_completion":
      continue
    task_id = completion.get("task_id")
    if not isinstance(task_id, str) or not task_id:
      continue
    try:
      envelope = AgentCompletionEnvelope.model_validate(
        completion.get("envelope")
      )
    except Exception:
      continue
    handle_and_grant = _completion_handle_and_grant(envelope)
    if handle_and_grant is None:
      continue
    content, grant = handle_and_grant
    if (
      content.content_id != content_id
      or grant.grant_id != read_grant_id
      or grant.content_id != content_id
      or grant.scope != "direct_parent"
      or grant.principal_id != principal_id
    ):
      continue
    registration_entries, _ = await durable_log.query(
      event_types={"task_registered"},
      contains_text=task_id,
      order="desc",
    )
    registrations = [
      item.event
      for item in registration_entries
      if item.event.get("task_id") == task_id
    ]
    if len(registrations) != 1:
      continue
    registration = registrations[0]
    if (
      registration.get("parent_session_id") != gateway_session_id
      or _workflow_owned(registration, registration.get("metadata"))
    ):
      continue
    task_result = await _durable_task_result(
      durable_log,
      task_id=task_id,
      expected_ref=envelope.task_result_ref,
    )
    if task_result is None:
      continue
    value_kind = _result_content_kind(task_result, content)
    if value_kind is None:
      continue
    candidates.append(_AuthorizedResultContent(
      handle=content,
      task_result=task_result,
      value_kind=value_kind,
    ))

  if not candidates:
    return None, {
      "code": "result_content_unavailable",
      "message": (
        "No authorized ordinary agent completion grants this content read"
      ),
    }
  canonical = {
    json.dumps(
      {
        "handle": candidate.handle.model_dump(mode="json"),
        "task_result_ref": TaskResultRef.from_result(
          candidate.task_result
        ).model_dump(mode="json"),
        "value_kind": candidate.value_kind,
      },
      sort_keys=True,
      separators=(",", ":"),
    )
    for candidate in candidates
  }
  if len(canonical) != 1:
    return None, {
      "code": "result_content_unavailable",
      "message": "Conflicting durable content handles exist for this grant",
    }
  return candidates[0], None


async def _durable_task_result(
  durable_log: Any,
  *,
  task_id: str,
  expected_ref: TaskResultRef,
) -> TaskResult | None:
  completion_entries, _ = await durable_log.query(
    event_types={"task_completed"},
    contains_text=task_id,
    order="desc",
  )
  results: list[TaskResult] = []
  for entry in completion_entries:
    event = entry.event
    if event.get("type") != "task_completed" or event.get("task_id") != task_id:
      continue
    try:
      task_result = TaskResult.model_validate(event.get("result"))
    except Exception:
      continue
    if (
      task_result.attempt.physical_task_id != task_id
      or TaskResultRef.from_result(task_result) != expected_ref
    ):
      continue
    results.append(task_result)
  if not results:
    return None
  canonical = {
    json.dumps(
      result.model_dump(mode="json"),
      sort_keys=True,
      separators=(",", ":"),
    )
    for result in results
  }
  if len(canonical) != 1:
    return None
  return results[0]


def _result_content_kind(
  result: TaskResult,
  source: ContentHandle,
) -> Literal["terminal_narrative", "projection"] | None:
  if result.values.terminal_narrative == source:
    return "terminal_narrative"
  projection = result.values.projection
  if (
    projection is not None
    and projection.content == source
    and projection.inline_view is not None
  ):
    return "projection"
  # NamedArtifact currently carries no byte source or registered content-store
  # identity. Completion publication therefore rejects artifact-only handles
  # instead of minting a grant that this reader cannot honor.
  return None


def _materialize_exact_content(
  authorized: _AuthorizedResultContent,
  *,
  workspace_dir: Any,
) -> str:
  handle = authorized.handle
  if authorized.value_kind == "terminal_narrative":
    text = read_final_narrative_by_content_handle(
      workspace_dir=workspace_dir,
      content=handle,
    )
    raw = text.encode(handle.encoding or "utf-8")
  else:
    projection = authorized.task_result.values.projection
    if projection is None or projection.inline_view is None:
      raise ValueError("canonical projection content is unavailable")
    raw = canonical_json_bytes(projection.inline_view)
    text = raw.decode("utf-8")
  if (
    len(raw) != handle.content_bytes
    or hashlib.sha256(raw).hexdigest() != handle.content_sha256
    or handle.content_id != f"sha256:{handle.content_sha256}"
    or (
      handle.content_chars is not None
      and len(text) != handle.content_chars
    )
  ):
    raise ValueError("materialized result content conflicts with its handle")
  return text


def _completion_handle_and_grant(
  envelope: AgentCompletionEnvelope,
) -> tuple[ContentHandle, ContentReadGrant] | None:
  materialization = envelope.parent_materialization
  if isinstance(
    materialization,
    (ResultHandle, AuthoredSummaryWithResultHandle),
  ):
    return materialization.source, materialization.read_grant
  return None


def _workflow_owned(registration: Mapping[str, Any], metadata: Any) -> bool:
  return (
    registration.get("workflow_run_id") is not None
    or (
      isinstance(metadata, Mapping)
      and metadata.get(WORKFLOW_TASK_METADATA_KEY) is not None
    )
  )


def _page_result_content(
  *,
  authorized: _AuthorizedResultContent,
  text: str,
  after_char: int,
) -> dict[str, Any]:
  if after_char > len(text):
    raise ValueError("after_char is outside the result content")
  content = json_bounded_text_prefix(
    text[after_char:],
    max_bytes=AGENT_RESULT_CONTENT_CHUNK_MAX_BYTES,
  )
  delivered_end = after_char + len(content)
  end = delivered_end == len(text)
  page: dict[str, Any] = {
    "kind": "agent_result_content",
    "value_kind": authorized.value_kind,
    "content_id": authorized.handle.content_id,
    "contract": authorized.handle.contract.model_dump(mode="json"),
    "media_type": authorized.handle.media_type,
    "encoding": authorized.handle.encoding,
    "after_char": after_char,
    "content": content,
    "content_sha256": authorized.handle.content_sha256,
    "content_chars": authorized.handle.content_chars,
    "content_bytes": authorized.handle.content_bytes,
    "next_cursor": None if end else {"after_char": delivered_end},
    "end": end,
  }
  if len(
    json.dumps(
      page,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
    ).encode("utf-8")
  ) > AGENT_RESULT_CONTENT_PAGE_MAX_BYTES:
    raise ValueError("agent result content page exceeds its byte limit")
  return page


__all__ = [
  "AGENT_RESULT_CONTENT_CHUNK_MAX_BYTES",
  "AGENT_RESULT_CONTENT_PAGE_MAX_BYTES",
  "AGENT_RESULT_CONTENT_MAX_SEQUENCE",
  "make_get_agent_result_content_handler",
  "make_get_agent_result_content_tool_def",
]
