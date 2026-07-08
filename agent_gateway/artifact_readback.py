"""Synthesize artifact-surface events for stored-artifact readbacks.

Lives in agent_gateway so both tool loops can share it: the SDK runner
(runner_tool_execution) that serves the interactive channels, and the direct-API
loop in api/agent/shared/tool_handlers/chat_streaming.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


READBACK_TOOLS = frozenset({"get_skill_artifact"})


def _clean_str(value: Any) -> str:
  return value.strip() if isinstance(value, str) else ""


def readback_artifact_ready_event(
  tool_name: str,
  result: Any,
  tool_call_id: Any,
) -> Optional[Dict[str, Any]]:
  """Synthesize an artifact_ready surface event for a stored-artifact readback.

  Emitted only for successful ``get_skill_artifact`` results so the taskpane can
  render the artifact the user asked to see. ``origin: "readback"`` lets the
  frontend distinguish user-requested surfaces (which pull tab focus) from fresh
  runs and synthetic rehydrates. Returns None for everything else — a readback
  must never break the tool loop.
  """
  if tool_name not in READBACK_TOOLS or not isinstance(result, dict):
    return None
  if result.get("status") != "success":
    return None
  artifact = result.get("artifact")
  if not isinstance(artifact, dict):
    return None
  ticker = _clean_str(result.get("ticker"))
  skill = _clean_str(result.get("skill"))
  artifact_id = _clean_str(result.get("artifact_id"))
  if not ticker or not skill or not artifact_id:
    return None
  # source_path is deliberately ignored: on risk-module readbacks it can point
  # at notes/skills markdown rather than the JSON sidecar.
  artifact_path = _clean_str(artifact.get("artifact_path")) or (
    f"artifacts/{ticker}/{skill}/{artifact_id}.json"
  )
  return {
    "type": "artifact_ready",
    "skill_run_id": f"readback-{_clean_str(tool_call_id) or 'unknown'}",
    "ticker": ticker,
    "skill": skill,
    "artifact_id": artifact_id,
    "artifact_path": artifact_path,
    "binary_artifact_path": None,
    "contract_name": _clean_str(artifact.get("contract_name")),
    "data_source": _clean_str(artifact.get("data_source")) or "live",
    "ts": time.time(),
    "origin": "readback",
    "scope": "ticker",
    "portfolio_id": None,
  }


__all__ = [
  "READBACK_TOOLS",
  "readback_artifact_ready_event",
]
