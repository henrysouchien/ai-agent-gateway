"""Deterministic sizing for values serialized into model-bound tool results."""

from __future__ import annotations

import hashlib
import json
from typing import Any


ERROR_ENVELOPE_WIRE_CAP_BYTES = 4 * 1024
ERROR_PREVIEW_MAX_CHARS = 160
ERROR_PREVIEW_PREFIX_CHARS = 64
ERROR_PREVIEW_DIGEST_HEX_CHARS = 16


def serialize_model_bound_result(value: Any) -> str:
  """Mirror the JSON string placed in a successful tool-result content field."""

  return json.dumps(value, default=str)


def model_bound_wire_size(value: Any) -> int:
  """Return the exact UTF-8 byte size of the model-bound serialized result."""

  return len(serialize_model_bound_result(value).encode("utf-8"))


def model_bound_result_fits(value: Any, *, max_bytes: int) -> bool:
  """Apply the gateway convention that a non-positive cap is unbounded."""

  return max_bytes <= 0 or model_bound_wire_size(value) <= max_bytes


def canonical_error_value_text(value: Any) -> str:
  """Serialize one diagnostic value deterministically for safe display."""

  try:
    return json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      allow_nan=False,
    )
  except (TypeError, ValueError):
    type_ref = f"{type(value).__module__}.{type(value).__qualname__}"
    return json.dumps(
      {"unserializable_type": type_ref},
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      allow_nan=False,
    )


def bounded_error_preview(value: Any) -> str:
  """Return a deterministic diagnostic preview with truncation evidence."""

  serialized = canonical_error_value_text(value)
  if len(serialized) <= ERROR_PREVIEW_MAX_CHARS:
    return serialized
  encoded = serialized.encode("utf-8")
  short_digest = hashlib.sha256(encoded).hexdigest()[
    :ERROR_PREVIEW_DIGEST_HEX_CHARS
  ]
  suffix = (
    f"<truncated>;chars={len(serialized)};bytes={len(encoded)};"
    f"sha256={short_digest}"
  )
  preview = f"{serialized[:ERROR_PREVIEW_PREFIX_CHARS]}{suffix}"
  if len(preview) > ERROR_PREVIEW_MAX_CHARS:
    raise AssertionError("bounded error preview metadata exceeds its cap")
  return preview


__all__ = [
  "ERROR_ENVELOPE_WIRE_CAP_BYTES",
  "ERROR_PREVIEW_DIGEST_HEX_CHARS",
  "ERROR_PREVIEW_MAX_CHARS",
  "ERROR_PREVIEW_PREFIX_CHARS",
  "bounded_error_preview",
  "canonical_error_value_text",
  "model_bound_result_fits",
  "model_bound_wire_size",
  "serialize_model_bound_result",
]
