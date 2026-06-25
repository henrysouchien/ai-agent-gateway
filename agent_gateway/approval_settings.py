from __future__ import annotations

import logging
import os


log = logging.getLogger("agent_gateway.approval_settings")

APPROVAL_WAIT_SECONDS_ENV = "GATEWAY_APPROVAL_WAIT_SECONDS"
# Default sized to the ~5-minute trade-preview lifetime, minus headroom for the
# preview->execute dispatch latency: a human-in-the-loop approval window should
# fit *within* one preview so the user can decide without the preview expiring
# mid-wait and forcing a re-preview, and so an approval granted at the end of the
# window doesn't then execute against a just-expired preview. 270s leaves ~30s of
# margin under the 300s preview TTL. 120s was too short for a momentary step-away
# and caused preview churn on irreversible trade approvals.
# This is only the global ceiling. Trade execution approvals are further capped
# against the preview's remaining TTL by `effective_trade_approval_expiry_seconds`.
DEFAULT_APPROVAL_WAIT_SECONDS = 270.0
MIN_APPROVAL_WAIT_SECONDS = 10.0
# Ceiling raised so operators can extend the window for non-time-bound approvals
# (the per-call wait is still further capped by the approval's own expiry).
MAX_APPROVAL_WAIT_SECONDS = 1800.0


def approval_wait_seconds() -> float:
  raw_value = os.getenv(APPROVAL_WAIT_SECONDS_ENV, str(int(DEFAULT_APPROVAL_WAIT_SECONDS))).strip()
  if not raw_value:
    return DEFAULT_APPROVAL_WAIT_SECONDS
  try:
    value = float(raw_value)
  except (TypeError, ValueError):
    log.warning(
      "Invalid %s=%r; using default %.0f",
      APPROVAL_WAIT_SECONDS_ENV,
      raw_value,
      DEFAULT_APPROVAL_WAIT_SECONDS,
    )
    return DEFAULT_APPROVAL_WAIT_SECONDS
  return min(MAX_APPROVAL_WAIT_SECONDS, max(MIN_APPROVAL_WAIT_SECONDS, value))


def read_approval_wait_seconds() -> float:
  return approval_wait_seconds()


__all__ = [
  "APPROVAL_WAIT_SECONDS_ENV",
  "DEFAULT_APPROVAL_WAIT_SECONDS",
  "MAX_APPROVAL_WAIT_SECONDS",
  "MIN_APPROVAL_WAIT_SECONDS",
  "approval_wait_seconds",
  "read_approval_wait_seconds",
]
