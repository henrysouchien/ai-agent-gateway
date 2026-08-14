from __future__ import annotations

import hashlib

from agent_gateway.model_bound_wire import (
  ERROR_PREVIEW_DIGEST_HEX_CHARS,
  ERROR_PREVIEW_MAX_CHARS,
  ERROR_PREVIEW_PREFIX_CHARS,
  bounded_error_preview,
  canonical_error_value_text,
)


def test_short_error_preview_is_exact_canonical_unicode_json() -> None:
  value = {"雪": "🙂", "a": [2, 1]}

  assert canonical_error_value_text(value) == '{"a":[2,1],"雪":"🙂"}'
  assert bounded_error_preview(value) == '{"a":[2,1],"雪":"🙂"}'


def test_bounded_error_preview_is_canonical_deterministic_and_self_describing() -> None:
  pathological = {
    "z": [
      {"雪🙂": "e\u0301漢字🙂" * 80, "control": "\u0001\n"},
      {"nested": ["🧮" * 70, {"é": "値" * 90}]},
    ],
    "a": "先頭" * 50,
  }
  serialized = canonical_error_value_text(pathological)
  encoded = serialized.encode("utf-8")
  digest = hashlib.sha256(encoded).hexdigest()[
    :ERROR_PREVIEW_DIGEST_HEX_CHARS
  ]
  expected = (
    serialized[:ERROR_PREVIEW_PREFIX_CHARS]
    + f"<truncated>;chars={len(serialized)};bytes={len(encoded)};"
    + f"sha256={digest}"
  )

  observed = tuple(bounded_error_preview(pathological) for _ in range(5))

  assert observed == (expected,) * 5
  assert len(expected) <= ERROR_PREVIEW_MAX_CHARS
  assert expected.startswith(serialized[:ERROR_PREVIEW_PREFIX_CHARS])
  assert "<truncated>" in expected
  assert len(encoded) > len(serialized)
