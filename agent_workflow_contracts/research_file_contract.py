"""Canonical research-file identity contract.

Two facts live here, and nowhere else.

`is_research_file_id` is the one expression of what a research file id IS — a
non-boolean integer in `1 .. MAX_RESEARCH_FILE_ID` — the rule this repository
already published as data (`positive-signed-64-bit-integer/v1`) but had never
published as code, so six boundaries hand-wrote it and disagreed three ways.

`format_research_file_id_token` is the one spelling the runtime uses to show a
run which research file it is bound to. It is write-only toward the model: the
binding itself is a typed scalar decided once at dispatch, and free text is not
a channel for it, so nothing in this tree parses that token back out.

Sibling of `ticker_contract`; stdlib-only, importable from both `api/` and
`agent_gateway/`.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from typing import TypeIs


MAX_RESEARCH_FILE_ID = (1 << 63) - 1


def is_research_file_id(value: object) -> TypeIs[int]:
  """Whether ``value`` is a research file id.

  The one expression of the rule this repository already published as data
  (`positive-signed-64-bit-integer/v1`): a non-boolean integer in
  ``1 .. MAX_RESEARCH_FILE_ID``. Every boundary that accepts a research file id
  asks this; each raises its own boundary-appropriate error.
  """
  return (
    not isinstance(value, bool)
    and isinstance(value, Integral)
    and 1 <= int(value) <= MAX_RESEARCH_FILE_ID
  )


def format_research_file_id_token(research_file_id: int) -> str:
  """The one spelling the runtime uses to show a run its binding."""
  if not is_research_file_id(research_file_id):
    raise ValueError("research file id must be a non-boolean integer in the supported range")
  return f"RESEARCH_FILE_ID={int(research_file_id)}"


__all__ = [
  "MAX_RESEARCH_FILE_ID",
  "format_research_file_id_token",
  "is_research_file_id",
]
