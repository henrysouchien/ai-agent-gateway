"""Risk-owned commercial-authority feed decoding."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from jsonschema.exceptions import ValidationError

from hank_contracts.commercial_authority import (
  _VALIDATOR,
  CommercialAuthorityInvalidation,
  DecodedCommercialAuthorityFeedPage,
)



def decode_commercial_authority_feed_v1(
  page: Any,
  *,
  cursor: int,
  expected_environment: str | None,
) -> DecodedCommercialAuthorityFeedPage:
  """Validate and decode a complete page without mutating consumer state."""

  try:
    _VALIDATOR.validate(page)
  except ValidationError as error:
    raise ValueError("commercial invalidation feed schema is invalid") from error

  next_sequence = page["next_sequence"]
  high_water_sequence = page["high_water_sequence"]
  if next_sequence < cursor or high_water_sequence < next_sequence:
    raise ValueError("commercial invalidation page is invalid")

  previous = cursor
  invalidations: list[CommercialAuthorityInvalidation] = []
  for raw in page["events"]:
    if (
      expected_environment is not None
      and raw["environment"] != expected_environment
    ):
      raise ValueError("commercial invalidation environment is invalid")
    occurred_at_text = raw["occurred_at"]
    if occurred_at_text.endswith(("Z", "z")):
      occurred_at_text = occurred_at_text[:-1] + "+00:00"
    try:
      occurred_at = datetime.fromisoformat(occurred_at_text)
    except ValueError as error:
      raise ValueError(
        "commercial invalidation occurrence time is invalid"
      ) from error
    if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
      raise ValueError("commercial invalidation occurrence time is invalid")

    sequence = raw["sequence_id"]
    if sequence <= previous:
      raise ValueError("commercial invalidation sequence is invalid")
    try:
      invalidation = CommercialAuthorityInvalidation(
        kind=raw["kind"],
        commercial_account_id=raw["commercial_account_id"],
        entitlement_revision=raw["entitlement_revision"],
        context_id=UUID(raw["context_id"]) if raw["context_id"] else None,
        token_id=UUID(raw["token_id"]) if raw["token_id"] else None,
      )
    except (TypeError, ValueError) as error:
      raise ValueError("commercial invalidation event is invalid") from error
    invalidations.append(invalidation)
    previous = sequence

  events = page["events"]
  if (events and previous != next_sequence) or (
    not events and next_sequence != cursor
  ):
    raise ValueError("commercial invalidation page cursor mismatch")
  if not events and cursor < high_water_sequence:
    raise ValueError("commercial invalidation feed made no catch-up progress")
  return DecodedCommercialAuthorityFeedPage(
    invalidations=tuple(invalidations),
    next_sequence=next_sequence,
    high_water_sequence=high_water_sequence,
  )


__all__ = [
  "decode_commercial_authority_feed_v1",
]
