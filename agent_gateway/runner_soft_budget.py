"""The one soft containment guard for delegated runs.

A delegated run (a ``run_agent`` child, a learning fork, an autonomous skill
child) can loop with nobody watching, so it meets exactly one runaway guard,
and this module owns it. The guard is a ladder, never a stop at a number:

1. **steer** at the threshold: a wrap-up notice naming the requests made and
   the threshold rides the child's next turn; the child keeps its tools and
   keeps working.
2. **final** at 1.5x the threshold: the loop opens its forced-final delivery
   epoch, whose turns carry the synthesis reminder, so the partial findings
   come back as an ordinary result.
3. **abort** only when the child has still not finished ``grace_turns`` turns
   (or continuation segments) after the forced final: the run ends
   ``interrupted`` with the counters on the event. This is the last resort.

The abort stays by ruling (Henry, 2026-10-06, rule 8 of
``docs/engineering/fail-closed-local-guards.md``): it is the **runaway bound on
an unwatched process**, the containment layer's own, not a cap on valid work.
It is reached only after the run declined two chances to answer, and nothing
else in the loop ends such a run: the forced-final epoch keeps running once
its credits are spent (``runner_run_loop._run_chat_loop``), and a child's
``timeout`` is a separate owner. Deleting this stage makes a child that never
answers unbounded.

The first two stages return a result; only the third ends the run. The
threshold has one source: ``SOFT_TURN_BUDGET``, keyed by run kind with a
``default`` (a ``run_agent``/skill child and a learning fork are kinds; nothing
a run declares feeds it). ``HANK_SOFT_TURN_BUDGET`` can only lower it and ``0``
disables the guard. A primary run has no ladder. The ladder counts provider
requests (completed turns plus completed ``max_tokens`` continuation segments),
so a child whose provider never completes a turn still meets every stage;
cost is a measurement on the run record, never a rung. Shape and rationale:
``docs/engineering/soft-budget-containment-reference.md`` (oh-my-pi's subagent
soft request budget).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

#: Bundled steer thresholds, in turns, keyed by run kind; a kind with no entry
#: resolves to ``default``.
SOFT_TURN_BUDGET: dict[str, int] = {"default": 100, "autonomous": 100, "fork": 10}
SOFT_BUDGET_FINAL_FACTOR = 1.5
#: Turns the child may run after the forced final before the run is aborted.
SOFT_BUDGET_GRACE_TURNS = 2
SOFT_TURN_BUDGET_ENV = "HANK_SOFT_TURN_BUDGET"

#: The wrap-up guidance the steer carries after its numbers
#: (``soft_budget_steer_notice``).
SOFT_BUDGET_STEER_NOTICE = (
  "Start wrapping up: finish the step you are on, avoid opening new lines of "
  "work, and move toward your final answer from the evidence already "
  "produced. Your tools remain available."
)


def configured_soft_turn_budget() -> int | None:
  """``HANK_SOFT_TURN_BUDGET`` as an int; unset or unreadable is ``None``."""

  raw = os.environ.get(SOFT_TURN_BUDGET_ENV, "").strip()
  if not raw:
    return None
  try:
    return int(raw)
  except ValueError:
    logger.warning(
      "%s=%r is not an integer; the bundled soft turn budget applies",
      SOFT_TURN_BUDGET_ENV,
      raw,
    )
    return None


def resolve_soft_turn_budget(kind: str, configured: int | None) -> int | None:
  """Resolve a run kind's steer threshold in turns; ``None`` means no ladder.

  The table entry for ``kind`` (else ``default``) is the threshold;
  ``configured`` can only lower it, and ``0`` (or less) disables the guard.
  """

  resolved = SOFT_TURN_BUDGET.get(kind, SOFT_TURN_BUDGET["default"])
  if configured is None:
    return resolved
  if configured <= 0:
    return None
  return min(resolved, configured)


@dataclass(frozen=True)
class SoftBudgetAction:
  """One ladder stage firing.

  ``observed`` is the provider requests completed when the stage fired
  (turns plus continuation segments); ``threshold`` is always the steer
  threshold. ``kind`` is always ``"turns"`` and stays on the event for the
  wire.
  """

  kind: Literal["turns"]
  stage: Literal["steer", "final", "abort"]
  observed: float
  threshold: float


def soft_budget_steer_notice(action: SoftBudgetAction) -> str:
  """The steer text for one ``steer`` action: the ladder's ``observed`` and
  ``threshold``, then the wrap-up guidance."""

  return (
    f"This run has made {action.observed:.0f} provider requests (turns plus "
    f"continuation segments); its expected share is {action.threshold:.0f}. "
    f"{SOFT_BUDGET_STEER_NOTICE}"
  )


class SoftBudgetLadder:
  """Steer, then force a final report, then abort after grace.

  ``observe`` is called before every provider request. Each stage fires at
  most once, and the one ``abort`` fires when ``grace_turns`` requests have
  run since the final.
  """

  def __init__(
    self,
    *,
    turn_threshold: int | None,
    grace_turns: int = SOFT_BUDGET_GRACE_TURNS,
  ) -> None:
    self._threshold = turn_threshold
    self._final_threshold = (
      None
      if turn_threshold is None
      else math.ceil(turn_threshold * SOFT_BUDGET_FINAL_FACTOR)
    )
    self.grace_turns = grace_turns
    self._steered = False
    self._final_requests: int | None = None
    self._segments = 0
    self._aborted = False

  def observe(
    self,
    *,
    turn_count: int,
    continuing_segment: bool = False,
  ) -> SoftBudgetAction | None:
    """``turn_count`` is the turns the loop has counted; on a
    ``continuing_segment`` (a request extending the current turn past
    ``max_tokens``) that already includes the turn being continued, whose
    earlier segments are the requests this ladder adds."""
    requests = turn_count + self._segments
    self._segments += int(continuing_segment)
    threshold = self._threshold
    final_threshold = self._final_threshold
    if self._aborted or threshold is None or final_threshold is None:
      return None

    def action(stage: Literal["steer", "final", "abort"]) -> SoftBudgetAction:
      return SoftBudgetAction(
        kind="turns",
        stage=stage,
        observed=float(requests),
        threshold=float(threshold),
      )

    if self._final_requests is not None:
      if requests - self._final_requests < self.grace_turns:
        return None
      self._aborted = True
      return action("abort")
    if requests >= final_threshold:
      self._final_requests = requests
      return action("final")
    if not self._steered and requests >= threshold:
      self._steered = True
      return action("steer")
    return None


__all__ = [
  "SOFT_BUDGET_FINAL_FACTOR",
  "SOFT_BUDGET_GRACE_TURNS",
  "SOFT_BUDGET_STEER_NOTICE",
  "SOFT_TURN_BUDGET",
  "SOFT_TURN_BUDGET_ENV",
  "SoftBudgetAction",
  "SoftBudgetLadder",
  "configured_soft_turn_budget",
  "resolve_soft_turn_budget",
  "soft_budget_steer_notice",
]
