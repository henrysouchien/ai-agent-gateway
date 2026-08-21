"""Canonical research-file identity contract for agent dispatch contexts.

One grammar, one extractor, one token spelling. Dispatch contexts are free text
carried across a real process boundary (the child's prompt), so the research
file a dispatch is bound to is stated *in* that text. Every producer must spell
it with `format_research_file_id_token`, and every consumer must read it with
this module's regex, so that "does this context already name a research file?"
has exactly one answer everywhere.

The key may be quoted because the same fact also arrives serialized — a typed
handoff payload embeds it as `{"research_file_id": 1, ...}`. A grammar that only
recognized the bare prose forms reported zero tokens for such a context, so a
dispatcher treated an explicitly bound context as ticker-only and appended a
second, different id.

Sibling of `ticker_contract`; stdlib-only, importable from both `api/` and
`agent_gateway/`.
"""

from __future__ import annotations

import re
from numbers import Integral
from typing import Any


RESEARCH_FILE_ID_RE = re.compile(
  r"\bresearch[_ ]file[_ ]id\b[\"']?\s*[:=]\s*(\d+)",
  re.IGNORECASE,
)


def research_file_id_values(text: object) -> tuple[int, ...]:
  """Every research file id stated in ``text``, in order of appearance."""
  if not isinstance(text, str):
    return ()
  return tuple(int(match.group(1)) for match in RESEARCH_FILE_ID_RE.finditer(text))


def extract_research_file_id(text: object, *, pattern: Any = RESEARCH_FILE_ID_RE) -> int | None:
  """The first research file id stated in ``text``, or ``None`` if it states none."""
  if not isinstance(text, str):
    return None
  match = pattern.search(text)
  if match is None:
    return None
  try:
    return int(match.group(1))
  except ValueError:
    return None


def format_research_file_id_token(research_file_id: int) -> str:
  """The one spelling a producer may use to state a research file id in context."""
  if isinstance(research_file_id, bool) or not isinstance(research_file_id, Integral):
    raise ValueError("research file id must be a non-boolean integer")
  return f"RESEARCH_FILE_ID={int(research_file_id)}"


__all__ = [
  "RESEARCH_FILE_ID_RE",
  "extract_research_file_id",
  "format_research_file_id_token",
  "research_file_id_values",
]
