"""Old, large data tool results leave the model-visible history for a stub (L4).

Owner design: ``docs/design/live-run-token-efficiency-2026-11-06.md`` (L4, P4,
condition 5). A tool result whose receipt registered a stub (the receipt
decides kind and size: ``runner_tool_execution``) is replaced in each provider
request, once enough assistant replies follow it, by that stub. The stub keeps
the head of the result, the identifiers a later quotation would need, and the
``spill_ref`` that ``tool_result_read`` resolves to the full retained content.
``messages`` and the durable log stay whole; only the request projection ages.

The boundary advances every ``step_replies`` replies, so between two steps the
projected prefix is byte-identical and the provider prefix cache breaks at most
once per step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

# Skill and methodology bodies are instructions the model keeps acting on; they
# never age, whatever their size (condition 5).
INSTRUCTION_RESULT_TOOLS: frozenset[str] = frozenset({"invoke_skill", "docs_fetch"})

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-:/\.]{4,79}|\d[\d,]{4,}(?:\.\d+)?")
_MIN_IDENTIFIER_CHARS = 5


@dataclass(frozen=True)
class ToolResultAging:
  after_replies: int
  min_chars: int
  step_replies: int


def h1_identifiers(content: str, *, limit: int = 400) -> list[str]:
  """Identifier-like tokens of ``content``: the longest ``limit``, longest first.

  A token is a string of at least five characters holding a digit, ``_``, ``:``
  or ``-``, or a number of at least five digits. Ties order lexicographically,
  so the same content always yields the same list.
  """

  tokens: set[str] = set()
  for token in _IDENTIFIER_RE.findall(content):
    token = token.rstrip(".")
    if token[0].isdigit() and "," in token:
      token = token.replace(",", "")
    if len(token) < _MIN_IDENTIFIER_CHARS:
      continue
    if any(char.isdigit() or char in "_:-" for char in token):
      tokens.add(token)
  return sorted(tokens, key=lambda token: (-len(token), token))[:limit]


def build_aging_stub(
  content: str,
  *,
  tool_name: str,
  spill_ref: str,
  head_chars: int = 1500,
) -> str:
  """The model-visible text that stands for an aged result; deterministic."""

  return json.dumps(
    {
      "_aged_tool_result": True,
      "tool_name": tool_name,
      "original_chars": len(content),
      "spill_ref": spill_ref,
      "message": (
        "This older result left the context; "
        "tool_result_read(spill_ref) returns it in full."
      ),
      "head": content[:head_chars],
      "identifiers": h1_identifiers(content),
    },
    ensure_ascii=False,
  )


def age_tool_results(
  messages: list[dict[str, Any]],
  *,
  aging: ToolResultAging,
  stubs: Mapping[str, str],
) -> list[dict[str, Any]]:
  """``messages`` with every registered, old-enough tool result swapped for its stub.

  Pure: neither ``messages`` nor its dicts change. A tool result ages when the
  stepped reply count (``replies_total`` rounded down to a multiple of
  ``step_replies``) exceeds the replies before its message by at least
  ``after_replies``. A message with nothing aged is returned as the same
  object, and ``messages`` itself when nothing ages.
  """

  if not stubs:
    return messages
  replies_total = sum(1 for message in messages if message.get("role") == "assistant")
  stepped = (replies_total // aging.step_replies) * aging.step_replies
  replies_before = 0
  aged: list[dict[str, Any]] | None = None
  for index, message in enumerate(messages):
    if message.get("role") == "assistant":
      replies_before += 1
      continue
    content = message.get("content")
    if stepped - replies_before < aging.after_replies or not isinstance(content, list):
      continue
    blocks = [
      {**block, "content": stubs[block["tool_use_id"]]}
      if (
        isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id") in stubs
      )
      else block
      for block in content
    ]
    if all(new is old for new, old in zip(blocks, content)):
      continue
    if aged is None:
      aged = list(messages)
    aged[index] = {**message, "content": blocks}
  return messages if aged is None else aged


__all__ = [
  "INSTRUCTION_RESULT_TOOLS",
  "ToolResultAging",
  "age_tool_results",
  "build_aging_stub",
  "h1_identifiers",
]
