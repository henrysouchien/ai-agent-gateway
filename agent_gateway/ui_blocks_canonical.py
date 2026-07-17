"""RFC 8785 canonicalization and identity for UI-block payloads."""

from __future__ import annotations

import hashlib
import math
from typing import Any


MAX_CANONICAL_PAYLOAD_BYTES = 32_768


class CanonicalizationError(ValueError):
  """Raised when a value cannot be represented by RFC 8785 JCS."""


def _scalar_string(value: str) -> str:
  """Combine explicit UTF-16 pairs and reject lone surrogate code points."""

  result: list[str] = []
  index = 0
  while index < len(value):
    code = ord(value[index])
    if 0xD800 <= code <= 0xDBFF:
      if index + 1 >= len(value):
        raise CanonicalizationError("lone high surrogate in string")
      low = ord(value[index + 1])
      if not 0xDC00 <= low <= 0xDFFF:
        raise CanonicalizationError("lone high surrogate in string")
      result.append(chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00))
      index += 2
      continue
    if 0xDC00 <= code <= 0xDFFF:
      raise CanonicalizationError("lone low surrogate in string")
    result.append(value[index])
    index += 1
  return "".join(result)


def _quote(value: str) -> str:
  value = _scalar_string(value)
  pieces = ['"']
  escapes = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
  }
  for character in value:
    escaped = escapes.get(character)
    if escaped is not None:
      pieces.append(escaped)
    elif ord(character) <= 0x1F:
      pieces.append(f"\\u{ord(character):04x}")
    else:
      pieces.append(character)
  pieces.append('"')
  return "".join(pieces)


def _shortest_digits(value: float) -> tuple[str, int]:
  """Return shortest round-trip digits and their decimal-point position.

  CPython's correctly-rounded shortest conversion supplies the digit sequence;
  placement is deliberately handled below using ECMA-262's rules rather than
  reusing Python's presentation.
  """

  text = repr(value).lower()
  if "e" in text:
    mantissa, exponent_text = text.split("e", 1)
    exponent = int(exponent_text)
    integer, _, fraction = mantissa.partition(".")
    digits = (integer + fraction).lstrip("0").rstrip("0") or "0"
    return digits, exponent + 1

  integer, dot, fraction = text.partition(".")
  combined = integer + (fraction if dot else "")
  leading_zeroes = len(combined) - len(combined.lstrip("0"))
  digits = combined.lstrip("0").rstrip("0") or "0"
  return digits, len(integer) - leading_zeroes


def _ecma_number(value: int | float) -> str:
  try:
    number = float(value)
  except (OverflowError, ValueError) as exc:
    raise CanonicalizationError("number is outside the IEEE-754 range") from exc
  if not math.isfinite(number):
    raise CanonicalizationError("NaN and Infinity are not permitted by JCS")
  if number == 0:
    return "0"

  sign = "-" if number < 0 else ""
  digits, n = _shortest_digits(abs(number))
  k = len(digits)
  if k <= n <= 21:
    rendered = digits + "0" * (n - k)
  elif 0 < n <= 21:
    rendered = digits[:n] + "." + digits[n:]
  elif -6 < n <= 0:
    rendered = "0." + "0" * (-n) + digits
  else:
    coefficient = digits if k == 1 else digits[0] + "." + digits[1:]
    exponent = n - 1
    rendered = coefficient + "e" + ("+" if exponent >= 0 else "") + str(exponent)
  return sign + rendered


def _utf16_sort_key(value: str) -> bytes:
  return _scalar_string(value).encode("utf-16-be")


def _serialize(value: Any) -> str:
  if value is None:
    return "null"
  if value is True:
    return "true"
  if value is False:
    return "false"
  if isinstance(value, (int, float)):
    return _ecma_number(value)
  if isinstance(value, str):
    return _quote(value)
  if isinstance(value, (list, tuple)):
    return "[" + ",".join(_serialize(item) for item in value) + "]"
  if isinstance(value, dict):
    if not all(isinstance(key, str) for key in value):
      raise CanonicalizationError("object member names must be strings")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
      scalar_key = _scalar_string(key)
      if scalar_key in normalized:
        raise CanonicalizationError("object contains duplicate Unicode member names")
      normalized[scalar_key] = item
    members = (
      _quote(key) + ":" + _serialize(normalized[key])
      for key in sorted(normalized, key=_utf16_sort_key)
    )
    return "{" + ",".join(members) + "}"
  raise CanonicalizationError(f"unsupported JCS value type: {type(value).__name__}")


def canonical_bytes(payload: Any) -> bytes:
  """Serialize *payload* to RFC 8785 canonical UTF-8 bytes."""

  return _serialize(payload).encode("utf-8")


def payload_too_large(payload: Any) -> bool:
  """Return whether the canonical submitted payload exceeds its byte cap."""

  return len(canonical_bytes(payload)) > MAX_CANONICAL_PAYLOAD_BYTES


def ui_blocks_id(session_id: str, turn_key: str, payload_submitted: Any) -> str:
  """Derive the stable logical-emission identity."""

  preimage = (
    session_id.encode("utf-8")
    + b":"
    + turn_key.encode("utf-8")
    + b":"
    + canonical_bytes(payload_submitted)
  )
  return "ub_" + hashlib.sha256(preimage).hexdigest()[:16]


__all__ = [
  "CanonicalizationError",
  "MAX_CANONICAL_PAYLOAD_BYTES",
  "canonical_bytes",
  "payload_too_large",
  "ui_blocks_id",
]
