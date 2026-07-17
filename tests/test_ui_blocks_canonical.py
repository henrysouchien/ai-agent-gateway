from __future__ import annotations

import math
import struct

import pytest

from agent_gateway.ui_blocks_canonical import (
  CanonicalizationError,
  canonical_bytes,
  payload_too_large,
  ui_blocks_id,
)


def test_rfc_8785_appendix_a_vector() -> None:
  payload = {
    "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27],
    "string": "€$\x0f\nA'B\"\\\"/",
    "literals": [None, True, False],
  }
  assert canonical_bytes(payload) == (
    b'{"literals":[null,true,false],"numbers":[333333333.3333333,'
    b'1e+30,4.5,0.002,1e-27],"string":"\xe2\x82\xac$\\u000f\\nA\'B\\\"\\\\\\\"/"}'
  )


@pytest.mark.parametrize(
  ("value", "expected"),
  [
    (-0.0, b"0"),
    (1e-7, b"1e-7"),
    (1e-6, b"0.000001"),
    (1e20, b"100000000000000000000"),
    (1e21, b"1e+21"),
    (2**53 - 1, b"9007199254740991"),
    (float(2**53), b"9007199254740992"),
    (float(2**53 + 1), b"9007199254740992"),
    (5e-324, b"5e-324"),
    (2.2250738585072014e-308, b"2.2250738585072014e-308"),
    (1.7976931348623157e308, b"1.7976931348623157e+308"),
  ],
)
def test_ecma_number_boundary_vectors(value: float, expected: bytes) -> None:
  assert canonical_bytes(value) == expected


def test_non_finite_numbers_and_lone_surrogates_are_typed_errors() -> None:
  for value in (math.nan, math.inf, -math.inf):
    with pytest.raises(CanonicalizationError):
      canonical_bytes(value)
  with pytest.raises(CanonicalizationError, match="surrogate"):
    canonical_bytes("\ud800")


def test_shortest_number_output_round_trips_for_spread_of_doubles() -> None:
  bit_patterns = [
    1,
    2,
    0x0010000000000000,
    0x3E70000000000000,
    0x3FF0000000000000,
    0x400921FB54442D18,
    0x4340000000000000,
    0x7FD0000000000000,
    0x7FEFFFFFFFFFFFFF,
  ]
  values = [struct.unpack(">d", bits.to_bytes(8, "big"))[0] for bits in bit_patterns]
  values += [-value for value in values]
  for value in values:
    assert float(canonical_bytes(value)) == value


def test_object_members_sort_by_utf16_code_units() -> None:
  payload = {"\ufb33": 7, "😀": 6, "€": 5, "ö": 4, "\x80": 3, "1": 2, "\r": 1}
  assert canonical_bytes(payload).decode() == (
    '{"\\r":1,"1":2,"\x80":3,"ö":4,"€":5,"😀":6,"דּ":7}'
  )
  assert canonical_bytes({"\ud83d\ude00": 1}) == canonical_bytes({"😀": 1})


def test_size_gate_accepts_exact_cap_and_rejects_next_byte() -> None:
  assert len(canonical_bytes({"x": "a" * 32760})) == 32768
  assert not payload_too_large({"x": "a" * 32760})
  assert len(canonical_bytes({"x": "a" * 32761})) == 32769
  assert payload_too_large({"x": "a" * 32761})


def test_id_derivation_inputs_and_emission_index_exclusion() -> None:
  payload = {"kind": "hank_ui_blocks.v1", "contract_version": 1, "blocks": []}
  identity = ui_blocks_id("session", "turn", payload)
  assert identity == "ub_570f9e0a9ee8eb72"
  assert ui_blocks_id("session-2", "turn", payload) != identity
  assert ui_blocks_id("session", "turn-2", payload) != identity
  assert ui_blocks_id("session", "turn", payload | {"lead_text": "x"}) != identity
  for _emission_index in (0, 1, 999):
    assert ui_blocks_id("session", "turn", payload) == identity
