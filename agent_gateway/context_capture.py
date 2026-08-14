"""Injected context-capture protocol and operational event helpers."""

from __future__ import annotations

import json
from typing import Any, Protocol


class ContextCapture(Protocol):
  def persist(
    self,
    *,
    surfaces: list[dict[str, Any]],
    rendered_system_prompt: str | list[tuple[str, bool]] | None,
  ) -> None:
    """Persist optional operational context without affecting execution."""


def _operational_surfaces(surfaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [
    {
      key: value
      for key, value in surface.items()
      if key not in {"content_hash", "source_hash", "variant_hash"}
    }
    for surface in surfaces
  ]


def canonical_manifest_digest(
  surfaces: list[dict[str, Any]],
  system_prompt_hash: str | None,
) -> str:
  """Return stable plain JSON for legacy callers that suppress repeat events."""
  del system_prompt_hash
  return json.dumps(
    _operational_surfaces(surfaces),
    sort_keys=True,
    ensure_ascii=True,
    separators=(",", ":"),
    default=str,
  )


def build_context_manifest_event(
  *,
  surfaces: list[dict[str, Any]],
  system_prompt_hash: str | None,
  session_id: str,
  request_id: str | None,
  turn: int | None,
  invocation: int | None = None,
) -> dict[str, Any]:
  del system_prompt_hash
  event: dict[str, Any] = {
    "type": "context_manifest",
    "session_id": session_id,
    "request_id": request_id,
    "turn": turn,
    "surfaces": _operational_surfaces(surfaces),
  }
  if invocation is not None:
    event["invocation"] = invocation
  return event


__all__ = [
  "ContextCapture",
  "build_context_manifest_event",
  "canonical_manifest_digest",
]
