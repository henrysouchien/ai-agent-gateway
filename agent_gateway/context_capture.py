"""Injected context-capture protocol and manifest helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol


class CaptureUnresolved(RuntimeError):
  """Raised when a context manifest cannot be made content-resolvable."""


class ContextCapture(Protocol):
  def persist(
    self,
    *,
    surfaces: list[dict[str, Any]],
    rendered_system_prompt: str | list[tuple[str, bool]] | None,
  ) -> str | None:
    """Persist the prompt and verify every surface hash is resolvable."""


def canonical_manifest_digest(
  surfaces: list[dict[str, Any]],
  system_prompt_hash: str | None,
) -> str:
  payload = {"surfaces": surfaces, "system_prompt_hash": system_prompt_hash}
  encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
  return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_context_manifest_event(
  *,
  surfaces: list[dict[str, Any]],
  system_prompt_hash: str | None,
  session_id: str,
  request_id: str | None,
  turn: int | None,
  invocation: int | None = None,
) -> dict[str, Any]:
  event: dict[str, Any] = {
    "type": "context_manifest",
    "session_id": session_id,
    "request_id": request_id,
    "turn": turn,
    "system_prompt_hash": system_prompt_hash,
    "surfaces": surfaces,
  }
  if invocation is not None:
    event["invocation"] = invocation
  return event


__all__ = [
  "CaptureUnresolved",
  "ContextCapture",
  "build_context_manifest_event",
  "canonical_manifest_digest",
]
