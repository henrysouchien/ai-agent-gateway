"""Canvas event construction outside the FMS ownership boundary."""

from __future__ import annotations

import time
from typing import Any


def emit_canvas_artifact_ready(
  *, artifact_id: str, skill_run_id: str, ticker: str | None, scope: str,
) -> dict[str, Any]:
  """Build the additive Canvas artifact-ready event.

  This intentionally lives outside ``api/fms`` while that tree has a degraded
  coordination write path. A future additive migration may place a delegating
  convention helper beside the HTML event without changing this event shape.
  """

  return {
    "type": "artifact_ready",
    "skill_run_id": skill_run_id,
    "ticker": ticker,
    "skill": "_canvas",
    "artifact_id": artifact_id,
    "artifact_path": f"artifacts/_canvas/{artifact_id}.json",
    "binary_artifact_path": f"artifacts/_canvas/{artifact_id}.bundle.js",
    "contract_name": "CanvasArtifact",
    "data_source": "live",
    "ts": time.time(),
    "scope": scope,
    "portfolio_id": None,
  }


__all__ = ["emit_canvas_artifact_ready"]
