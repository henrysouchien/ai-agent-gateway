"""Small, store-independent envelope for reviews of agent outputs."""

from __future__ import annotations

import time
from typing import Any, Mapping


DECISION_RECORD_SCHEMA_VERSION = 1


def build_decision_record(
  *,
  actor: str,
  verdict: str,
  object_kind: str,
  object_ref: Mapping[str, Any],
  reason: str | None = None,
  run: Mapping[str, Any] | None = None,
  decided_at: float | str | None = None,
) -> dict[str, Any]:
  """Describe a decision without constraining the owning seam's vocabulary."""
  record: dict[str, Any] = {
    "schema_version": DECISION_RECORD_SCHEMA_VERSION,
    "actor": actor,
    "verdict": verdict,
    "object_kind": object_kind,
    "object_ref": dict(object_ref),
    "reason": (reason.strip() or None) if reason is not None else None,
    "decided_at": time.time() if decided_at is None else decided_at,
  }
  if run:
    record["run"] = dict(run)
  return record


def run_pointer_from_context(tool_ctx: Any) -> dict[str, Any] | None:
  """Keep available trajectory join keys from an execution context."""
  pointer: dict[str, Any] = {}
  for key in ("session_id", "request_id", "skill_run_id", "control_run_id", "tool_call_id"):
    value = tool_ctx.get(key) if isinstance(tool_ctx, Mapping) else getattr(tool_ctx, key, None)
    if value is not None and value != "":
      pointer[key] = value
  return pointer or None
