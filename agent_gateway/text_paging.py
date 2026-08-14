from __future__ import annotations


def json_bounded_text_prefix(value: str, *, max_bytes: int) -> str:
  """Return the largest prefix whose JSON-string contribution fits."""

  if type(value) is not str:
    raise TypeError("paged text must be exact text")
  if type(max_bytes) is not int or max_bytes < 1:
    raise ValueError("max_bytes must be a positive integer")
  used_bytes = 0
  end = 0
  for character in value:
    codepoint = ord(character)
    if character in {'"', "\\"}:
      character_bytes = 2
    elif codepoint < 0x20:
      character_bytes = 2 if character in "\b\t\n\f\r" else 6
    else:
      character_bytes = len(character.encode("utf-8"))
    if used_bytes + character_bytes > max_bytes:
      break
    used_bytes += character_bytes
    end += 1
  return value[:end]


__all__ = ["json_bounded_text_prefix"]
