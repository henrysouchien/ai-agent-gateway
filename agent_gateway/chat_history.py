"""Display-only base-chat pages from the session's existing durable log."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, TypedDict

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .agent_session_log import AgentSessionLog, QueryCursor
from .session import GatewaySession


class ChatHistoryQuery(BaseModel):
  model_config = ConfigDict(extra="forbid")

  limit: int = Field(default=50, ge=1, le=200)
  before_seq: int | None = Field(default=None, ge=1)


class ChatHistoryEntry(TypedDict):
  seq: int
  role: Literal["user", "assistant"]
  content: str
  timestamp: float


class ChatHistoryPage(TypedDict):
  schema_version: Literal[1]
  entries: list[ChatHistoryEntry]
  next_before_seq: int | None


def _text_content(content: str | list[dict[str, Any]]) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return _text_blocks(content)
  raise TypeError("analyst content must be text or a content block list")


def _text_blocks(blocks: list[dict[str, Any]]) -> str:
  # Only text is displayable. In particular, thinking, tool calls/results,
  # images and structured/actionable blocks never cross this read surface.
  return "".join(block["text"] for block in blocks if block["type"] == "text")


async def read_chat_history(
  session: GatewaySession,
  query: ChatHistoryQuery,
) -> ChatHistoryPage:
  session_log = session.agent_session_log
  if session_log is None:
    path = session.resume_log_path
    if path is None:
      path = session.fork_source_log_path
    if path is None:
      return {"schema_version": 1, "entries": [], "next_before_seq": None}
    # Init's resume resolver already bound ownership. A read must not create an
    # active file or repair rotation metadata before the first resumed turn.
    session_log = await asyncio.to_thread(
      AgentSessionLog,
      path=path,
      _create_active_on_open=False,
      _repair_manifest_on_open=False,
    )

  raw_entries, cursor = await session_log.query(
    event_types={"user_message", "assistant_message"},
    role="writer",
    order="desc",
    limit=query.limit,
    cursor=(
      QueryCursor(after_seq=query.before_seq, direction="desc")
      if query.before_seq is not None
      else None
    ),
  )
  entries: list[ChatHistoryEntry] = []
  for entry in reversed(raw_entries):
    event = entry.event
    # Visibility belongs to this projection, not to a client-side event replay.
    if event.get("sub_agent_id") is not None or event.get("client_kind") == "operator":
      continue
    role: Literal["user", "assistant"] = (
      "user" if event["type"] == "user_message" else "assistant"
    )
    if role == "user" and "analyst_content" not in event:
      raise HTTPException(
        status_code=409,
        detail={
          "code": "history_analyst_content_unavailable",
          "seq": entry.seq,
        },
      )
    content = (
      _text_content(event["analyst_content"])
      if role == "user"
      else _text_blocks(event["content_blocks"])
    )
    if content:
      entries.append({
        "seq": entry.seq,
        "role": role,
        "content": content,
        "timestamp": entry.timestamp,
      })
  return {
    "schema_version": 1,
    "entries": entries,
    # Keep paging even when every raw record in a page is display-invisible.
    "next_before_seq": cursor.after_seq if cursor is not None else None,
  }
