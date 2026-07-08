from __future__ import annotations

import time
from typing import Any, Callable


async def emit_execution_audit(
  request: Any | None,
  raw_tool_args: dict[str, Any],
  *,
  approval_store: Any | None,
  outcome: str,
  error_summary: str | None = None,
) -> None:
  if request is None or approval_store is None:
    return
  emitter = getattr(approval_store, "audit_emitter", None)
  emit = getattr(emitter, "emit_execution_outcome", None) if emitter is not None else None
  if emit is None:
    return
  raw_args = dict(raw_tool_args)
  try:
    await emit(
      request=request,
      raw_tool_args=raw_args,
      outcome=outcome,
      error_summary=error_summary,
    )
  finally:
    raw_args.clear()
    del raw_args


def emit_approval_decided(
  event_log: Any | None,
  tool_call_id: str,
  tool_name: str,
  *,
  outcome: str,
  decision_source: str,
  allow_tool_type_applied: bool,
  time_fn: Callable[[], float] = time.time,
) -> None:
  if event_log is None:
    return
  event_log.append(
    {
      "type": "tool_approval_decided",
      "tool_call_id": tool_call_id,
      "tool_name": tool_name,
      "outcome": outcome,
      "decision_source": decision_source,
      "allow_tool_type_applied": allow_tool_type_applied,
      "ts": time_fn(),
    }
  )


__all__ = [
  "emit_approval_decided",
  "emit_execution_audit",
]
