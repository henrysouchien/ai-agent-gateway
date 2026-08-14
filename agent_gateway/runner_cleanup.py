from __future__ import annotations

from collections.abc import Iterator
from typing import Any


_CLEANUP_NOTE_PREFIXES = (
  "Child cleanup failed:",
  "Sub-agent cleanup failed:",
)


def cleanup_failure_detail(
  exc: BaseException,
  *,
  label: str = "Child cleanup failed",
) -> str:
  message = str(exc).strip() or type(exc).__name__
  return f"{label}: {type(exc).__name__}: {message}"


def attach_cleanup_failure(
  primary: BaseException,
  cleanup: BaseException,
  *,
  label: str = "Child cleanup failed",
) -> str:
  """Preserve ``primary`` while attaching one cleanup failure as diagnostics."""

  detail = cleanup_failure_detail(cleanup, label=label)
  try:
    primary.add_note(detail)
  except (AttributeError, TypeError):
    pass
  return detail


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
  seen: set[int] = set()
  current: BaseException | None = exc
  while current is not None and id(current) not in seen:
    seen.add(id(current))
    yield current
    current = current.__cause__ or current.__context__


def cleanup_failure_notes(exc: BaseException) -> tuple[str, ...]:
  """Return de-duplicated cleanup notes from an exception and its cause chain."""

  notes: list[str] = []
  for chained in _exception_chain(exc):
    raw_notes: Any = getattr(chained, "__notes__", ())
    if not isinstance(raw_notes, (list, tuple)):
      continue
    for raw_note in raw_notes:
      note = str(raw_note).strip()
      if (
        note
        and note.startswith(_CLEANUP_NOTE_PREFIXES)
        and note not in notes
      ):
        notes.append(note)
  return tuple(notes)
