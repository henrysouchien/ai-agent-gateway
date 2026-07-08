from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence


UI_BLOCKS_OPEN = ":::ui-blocks"
UI_BLOCKS_FALLBACK_NOTICE = "A generated visual element could not be rendered."

_LINE_CLOSE_RE = re.compile(r"^[ \t]*:::[ \t]*(?=\n|$)", re.MULTILINE)
_OPEN_TAIL_CHARS = len(UI_BLOCKS_OPEN) - 1


@dataclass(frozen=True)
class _FenceClose:
  start: int
  end: int


def _find_fence_close(text: str) -> _FenceClose | None:
  line_match = _LINE_CLOSE_RE.search(text, len(UI_BLOCKS_OPEN))
  if line_match is not None:
    return _FenceClose(start=line_match.start(), end=line_match.end())
  inline_index = text.find(":::", len(UI_BLOCKS_OPEN))
  if inline_index >= 0:
    return _FenceClose(start=inline_index, end=inline_index + 3)
  return None


def _payload_is_valid_ui_blocks(payload: str) -> bool:
  try:
    parsed = json.loads(payload.strip())
  except json.JSONDecodeError:
    return False
  return isinstance(parsed, list)


def _fallback_text() -> str:
  return f"\n\n{UI_BLOCKS_FALLBACK_NOTICE}\n\n"


class UIBlocksStreamFilter:
  """Strip malformed streamed :::ui-blocks fences without leaking raw JSON."""

  def __init__(self) -> None:
    self._buffer = ""
    self._inside_fence = False

  def feed(self, text: str) -> str:
    if not text:
      return ""
    self._buffer += text
    return self._drain(final=False)

  def finish(self) -> str:
    return self._drain(final=True)

  def _drain(self, *, final: bool) -> str:
    output: list[str] = []
    while self._buffer:
      if not self._inside_fence:
        open_index = self._buffer.find(UI_BLOCKS_OPEN)
        if open_index < 0:
          if final:
            output.append(self._buffer)
            self._buffer = ""
          elif len(self._buffer) > _OPEN_TAIL_CHARS:
            output.append(self._buffer[:-_OPEN_TAIL_CHARS])
            self._buffer = self._buffer[-_OPEN_TAIL_CHARS:]
          break
        if open_index:
          output.append(self._buffer[:open_index])
          self._buffer = self._buffer[open_index:]
        self._inside_fence = True

      close = _find_fence_close(self._buffer)
      if close is None:
        if final:
          output.append(_fallback_text())
          self._buffer = ""
          self._inside_fence = False
        break

      payload = self._buffer[len(UI_BLOCKS_OPEN):close.start]
      fence = self._buffer[:close.end]
      remainder = self._buffer[close.end:]
      output.append(fence if _payload_is_valid_ui_blocks(payload) else _fallback_text())
      self._buffer = remainder
      self._inside_fence = False

    return "".join(output)


def sanitize_ui_blocks_text(text: str) -> str:
  stream_filter = UIBlocksStreamFilter()
  return stream_filter.feed(text) + stream_filter.finish()


def sanitize_ui_blocks_content_blocks(content_blocks: Sequence[Any]) -> list[Any]:
  sanitized: list[Any] = []
  for block in content_blocks:
    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
      updated = dict(block)
      updated["text"] = sanitize_ui_blocks_text(updated["text"])
      sanitized.append(updated)
    else:
      sanitized.append(block)
  return sanitized
