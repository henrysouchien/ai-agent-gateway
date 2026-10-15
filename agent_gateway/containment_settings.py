"""The containment layer's own resource bounds, as one settings object.

``ContainmentSettings`` is the single owner of every bound rule 8 of
``docs/engineering/fail-closed-local-guards.md`` keeps (Henry's B1-B5 ruling,
``docs/closed/spawn-containment-bounds.md``) and of the context-pressure and
compaction thresholds. Each field names its oh-my-pi ``task.*`` counterpart.
The table of setting, env name, default and off value is
``docs/reference/agent-run-containment.md`` *Resource bounds: containment
settings*.

A process resolves it once at its start with
``resolve_containment_settings(os.environ)`` — the gateway at its composition
root, an autonomous child or a script at its own entry — and passes the object
down: admission reads it from ``AdmissionOwners.containment``, a runner from
its ``containment`` argument, and a child runner inherits its parent's. An
autonomous child process resolves the same values because every env name in
``CONTAINMENT_ENV`` is in its launch projection (``child_environment``).

An unreadable ``HANK_MAX_SPAWN_DEPTH`` is a startup error naming the setting:
admission's ``int()`` of it already raised, at spawn time. Every other
unreadable or out-of-range value is logged and its default applies, as an
unreadable ``HANK_SOFT_TURN_BUDGET`` always was; no other setting adds a
refusal.
"""

from __future__ import annotations

import logging
import math

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .runner_streaming import STREAM_THINKING_STALL_TIMEOUT

logger = logging.getLogger(__name__)


class ContainmentSettingError(ValueError):
  """One containment setting is unparseable or outside its range."""

  def __init__(self, setting: str, detail: str, *, env: str | None = None) -> None:
    self.setting = setting
    self.env = env
    source = f" ({env})" if env is not None else ""
    super().__init__(f"containment setting {setting}{source}: {detail}")


def _default_soft_turn_budget() -> Mapping[str, int]:
  return MappingProxyType({"default": 100, "autonomous": 100, "fork": 10})


@dataclass(frozen=True, slots=True)
class ContainmentSettings:
  """The ratified containment bounds; defaults are the ruled values."""

  max_spawn_depth: int = 2
  """B1. Admitted spawn hops below the root (root 0 -> lead 1 -> worker 2).
  The deepest tier's grant omits ``run_agent``; a spawn from a parent already at
  the bound is the one resource-class ``depth_exceeded``. Negative disables.
  omp ``task.maxRecursionDepth`` (default 2, negative disables)."""

  max_concurrent_children: int = 6
  """B2. Children one runner runs at once; waiters queue, never refuse. A child
  runner inherits the count, not the semaphore. At least 1; no disable value.
  It has no env name: the ruled single value is set only where the composition
  root constructs the settings (``SUB_AGENT_MAX_CONCURRENCY`` stays deleted).
  omp ``task.maxConcurrency`` (default 32; its ``0`` = unlimited is not
  adopted: the ruling keeps the queue)."""

  soft_turn_budget: Mapping[str, int] = field(default_factory=_default_soft_turn_budget)
  """B3. The soft ladder's steer threshold in provider requests, keyed by run
  kind; a kind with no entry resolves to ``default``. omp
  ``task.softRequestBudget`` (one number, 200)."""

  soft_turn_budget_cap: int | None = None
  """B3. Lowers every kind's threshold to at most this value; ``0`` or less
  disables the ladder; ``None`` leaves the table in force."""

  soft_budget_final_factor: float = 1.5
  """B3. The forced-final turn opens at this multiple of the threshold. omp
  forces the final at 1.5x ``task.softRequestBudget``."""

  soft_budget_grace_turns: int = 2
  """B3. Requests after the forced final before the run ends ``interrupted``
  (``soft_budget_abort``). omp's ``softRequestBudgetNotice`` is the steer."""

  child_silence_timeout_seconds: float = float(2 * STREAM_THINKING_STALL_TIMEOUT)
  """B4. The existing hung-child owner: a ``run_agent`` child or learning fork
  silent this long with no tool call in flight is cancelled and settles
  ``interrupted`` (``stalled``). Not a wall clock on valid work. ``0`` is the
  off value (omp's ``task.maxRuntimeMs`` shape, default 0 = none, though that
  is a wall clock and B4 adds none). Kept above the stream stall guard so the
  child's own guard fires first."""

  model_context_limit: int = 200_000
  """Context window assumed for a model that reports none."""

  context_warning_pct: int = 80
  """Context usage, in percent of the window, that logs a context warning."""

  context_pressure_reminder_pct: int = 60
  """Context usage that first adds the model-visible context-pressure reminder."""

  context_pressure_reminder_step_pct: int = 10
  """Further percentage points before the next reminder."""

  compaction_trigger_pct: int = 80
  """A configured compaction trigger is raised to this percent of a larger
  window (``effective_compaction_trigger``)."""

  def __post_init__(self) -> None:
    object.__setattr__(
      self, "soft_turn_budget", MappingProxyType(dict(self.soft_turn_budget)),
    )
    if "default" not in self.soft_turn_budget:
      raise ContainmentSettingError(
        "soft_turn_budget", "the table needs a 'default' entry",
      )
    for name in _MINIMUMS:
      problem = _range_problem(name, getattr(self, name))
      if problem is not None:
        raise ContainmentSettingError(name, problem, env=_ENV_BY_SETTING.get(name))

  def soft_turn_threshold(self, kind: str) -> int | None:
    """One run kind's steer threshold in requests; ``None`` means no ladder."""

    resolved = self.soft_turn_budget.get(kind, self.soft_turn_budget["default"])
    cap = self.soft_turn_budget_cap
    if cap is None:
      return resolved
    if cap <= 0:
      return None
    return min(resolved, cap)


_MINIMUMS: dict[str, int | float] = {
  "max_concurrent_children": 1,
  "soft_budget_final_factor": 1.0,
  "soft_budget_grace_turns": 0,
  "child_silence_timeout_seconds": 0.0,
  "model_context_limit": 1,
  "context_warning_pct": 1,
  "context_pressure_reminder_pct": 1,
  "context_pressure_reminder_step_pct": 1,
  "compaction_trigger_pct": 1,
}
_PERCENTAGES = (
  "context_warning_pct",
  "context_pressure_reminder_pct",
  "compaction_trigger_pct",
)


#: The env name each setting resolves from, and its parser. One table: the
#: autonomous child's launch projection carries every name in it.
CONTAINMENT_ENV: Mapping[str, tuple[str, Callable[[str], Any]]] = MappingProxyType({
  "max_spawn_depth": ("HANK_MAX_SPAWN_DEPTH", int),
  "soft_turn_budget_cap": ("HANK_SOFT_TURN_BUDGET", int),
  "soft_budget_final_factor": ("HANK_SOFT_BUDGET_FINAL_FACTOR", float),
  "soft_budget_grace_turns": ("HANK_SOFT_BUDGET_GRACE_TURNS", int),
  "child_silence_timeout_seconds": ("HANK_CHILD_SILENCE_TIMEOUT_SECONDS", float),
  "model_context_limit": ("HANK_MODEL_CONTEXT_LIMIT", int),
  "context_warning_pct": ("HANK_CONTEXT_WARNING_PCT", int),
  "context_pressure_reminder_pct": ("HANK_CONTEXT_PRESSURE_REMINDER_PCT", int),
  "context_pressure_reminder_step_pct": (
    "HANK_CONTEXT_PRESSURE_REMINDER_STEP_PCT", int,
  ),
  "compaction_trigger_pct": ("HANK_COMPACTION_TRIGGER_PCT", int),
})
_ENV_BY_SETTING = {name: env for name, (env, _parse) in CONTAINMENT_ENV.items()}
CONTAINMENT_ENV_NAMES = frozenset(_ENV_BY_SETTING.values())


def _range_problem(name: str, value: Any) -> str | None:
  minimum = _MINIMUMS.get(name)
  if value is None:
    return None
  if isinstance(value, float) and not math.isfinite(value):
    return f"{value!r} is not a finite number"
  if minimum is None:
    return None
  if value < minimum:
    return f"{value!r} is below the minimum {minimum!r}"
  if name in _PERCENTAGES and value > 100:
    return f"{value!r} is above 100 percent"
  return None


def resolve_containment_settings(environ: Mapping[str, str]) -> ContainmentSettings:
  """Read every set ``CONTAINMENT_ENV`` name; unset or blank keeps the default.

  Operator config inside the trust boundary is observed, never refused
  (docs/engineering/fail-closed-local-guards.md): an unreadable or
  out-of-range value, ``HANK_MAX_SPAWN_DEPTH`` included, logs one warning
  naming the setting, its env name and the value, and keeps that setting's
  default. A valid negative depth still disables the bound.
  """

  values: dict[str, Any] = {}
  for name, (env, parse) in CONTAINMENT_ENV.items():
    raw = str(environ.get(env, "")).strip()
    if not raw:
      continue
    try:
      value = parse(raw)
    except ValueError:
      problem = f"{raw!r} is not a valid {parse.__name__}"
    else:
      range_problem = _range_problem(name, value)
      if range_problem is None:
        values[name] = value
        continue
      problem = range_problem
    logger.warning(
      "containment setting %s (%s=%r): %s; the default applies",
      name, env, raw, problem,
    )
  return ContainmentSettings(**values)


__all__ = [
  "CONTAINMENT_ENV",
  "CONTAINMENT_ENV_NAMES",
  "ContainmentSettingError",
  "ContainmentSettings",
  "resolve_containment_settings",
]
