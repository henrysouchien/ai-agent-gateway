"""The containment layer's resource bounds, as a view over the run settings file.

``agent_gateway.run_settings`` owns every value, default and range (the
operator file ``~/.config/hank/run-settings.json``, override
``HANK_RUN_SETTINGS_PATH``, written with ``scripts/hank_settings.py``); this
file owns only how the gateway reads them. The table of setting, default and
range is ``docs/reference/agent-run-containment.md`` *Resource bounds:
containment settings*. Each setting names its oh-my-pi ``task.*`` counterpart
there.

``ContainmentSettings()`` is live: every attribute read resolves
``load_run_settings()`` (stamp-cached, one ``stat``), so a settings change is
in force at the next read with no restart. A malformed operator file raises
``RunSettingsError`` at that read. ``ContainmentSettings(**overrides)`` is
pinned: the packaged defaults plus the overrides, resolved once, never reading
the operator file (tests and callers that want a fixed bound use this form).

The gateway builds one live object at its composition root and passes it down:
admission reads it from ``AdmissionOwners.containment``, a runner from its
``containment`` argument, and a child runner inherits its parent's. An
autonomous child process reads the same operator file because
``HANK_RUN_SETTINGS_PATH`` is in its launch projection (``child_environment``).
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, overload

from .run_settings import (
  RunSettings,
  load_run_settings,
  pinned_run_settings,
)

_T = TypeVar("_T")


class _Setting(Generic[_T]):
  """One ``RunSettings`` field, resolved at each read of a live view."""

  def __init__(self, doc: str) -> None:
    self.__doc__ = doc
    self._name = ""

  def __set_name__(self, owner: type, name: str) -> None:
    self._name = name

  @overload
  def __get__(self, instance: None, owner: type) -> _Setting[_T]: ...

  @overload
  def __get__(self, instance: ContainmentSettings, owner: type) -> _T: ...

  def __get__(self, instance: ContainmentSettings | None, owner: type) -> Any:
    if instance is None:
      return self
    return getattr(instance._settings(), self._name)


class ContainmentSettings:
  """The containment bounds; live with no arguments, pinned with overrides."""

  __slots__ = ("_pinned",)

  def __init__(self, **overrides: Any) -> None:
    self._pinned: RunSettings | None = (
      pinned_run_settings(**overrides) if overrides else None
    )

  def _settings(self) -> RunSettings:
    return self._pinned if self._pinned is not None else load_run_settings()

  def __repr__(self) -> str:
    if self._pinned is None:
      return "ContainmentSettings(live)"
    return f"ContainmentSettings(pinned, {self._pinned.to_dict()!r})"

  max_spawn_depth = _Setting[int](
    """B1. Admitted spawn hops below the root (root 0 -> lead 1 -> worker 2).
    The deepest tier's grant omits ``run_agent``; a spawn from a parent already
    at the bound is the one resource-class ``depth_exceeded``. omp
    ``task.maxRecursionDepth`` (default 2)."""
  )
  max_concurrent_children = _Setting[int](
    """B2. Children one runner runs at once; waiters queue, never refuse. A
    child runner inherits the count, not the semaphore. At least 1. omp
    ``task.maxConcurrency`` (default 32; its ``0`` = unlimited is not adopted:
    the ruling keeps the queue)."""
  )
  writer_soft_turn_budget = _Setting[int | None](
    """B3. The soft ladder's steer threshold, in provider requests, for the
    root writer session; ``None`` leaves writer sessions without a ladder."""
  )
  child_soft_turn_budget = _Setting[int](
    """B3. The ladder's steer threshold for delegated (``sub_agent``) and
    ``autonomous`` runs. omp ``task.softRequestBudget`` (200)."""
  )
  fork_soft_turn_budget = _Setting[int](
    """B3. The ladder's steer threshold for learning forks."""
  )
  soft_budget_final_factor = _Setting[float](
    """B3. The forced-final turn opens at this multiple of the threshold. omp
    forces the final at 1.5x ``task.softRequestBudget``."""
  )
  soft_budget_grace_turns = _Setting[int](
    """B3. Requests after the forced final before the run ends ``interrupted``
    (``soft_budget_abort``). omp's ``softRequestBudgetNotice`` is the steer."""
  )
  child_silence_timeout_seconds = _Setting[float](
    """B4. The hung-child owner: a ``run_agent`` child or learning fork silent
    this long with no tool call in flight is cancelled and settles
    ``interrupted`` (``stalled``). Not a wall clock on valid work; ``0`` is the
    off value. Kept above the stream stall guard so the child's own guard
    fires first."""
  )
  repeat_tool_error_limit = _Setting[int | None](
    """The dispatcher's default repeat-tool-error bound for
    ``model_build_journey_measure.py``; the gateway does not count tool
    errors."""
  )
  model_context_limit = _Setting[int](
    """Context window assumed for a model that reports none."""
  )
  context_warning_pct = _Setting[int](
    """Context usage, in percent of the window, that logs a context warning."""
  )
  context_pressure_reminder_pct = _Setting[int](
    """Context usage that first adds the model-visible context-pressure
    reminder."""
  )
  context_pressure_reminder_step_pct = _Setting[int](
    """Further percentage points before the next reminder."""
  )
  compaction_trigger_pct = _Setting[int](
    """A configured compaction trigger is raised to this percent of a larger
    window (``effective_compaction_trigger``)."""
  )

  def soft_turn_threshold(self, kind: str) -> int | None:
    """One run kind's steer threshold in requests; ``None`` means no ladder."""

    return self._settings().soft_turn_threshold(kind)


__all__ = ["ContainmentSettings"]
