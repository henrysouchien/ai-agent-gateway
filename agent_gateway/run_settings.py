"""Run-and-loop diagnostic bounds: packaged defaults under one operator file.

Owner doc: ``docs/reference/agent-run-containment.md``. The bounds are the soft-turn ladder
thresholds per session kind, its final factor and grace turns, the child silence timeout,
spawn depth and concurrency, the context-pressure percentages and the repeat-tool-error
limit. This module is the only reader of the operator file (``~/.config/hank/run-settings.json``,
path override ``HANK_RUN_SETTINGS_PATH``) and the only writer of it (``set_operator_setting``,
reached by ``scripts/hank_settings.py set``). Every ``load_run_settings()`` re-stats the file
(``operator_settings.StampCache``), so a change is in force at the next read of a running
process. ``pinned_run_settings`` is the fixed form: packaged defaults plus overrides, never the
file. The file machinery is ``model_authority.operator_settings``, shared with
``spend_settings``; a bound here is a measurement an operator sets, never a refusal of an
artifact.
"""

from __future__ import annotations

import copy
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

from model_authority import operator_settings

SETTINGS_PATH_ENV: Final = "HANK_RUN_SETTINGS_PATH"
DEFAULT_SETTINGS_PATH: Final = "~/.config/hank/run-settings.json"
_DEFAULTS_PATH: Final = Path(__file__).resolve().parent / "run_settings.defaults.json"
_LABEL: Final = "run settings"


@dataclass(frozen=True)
class _Spec:
  integer: bool
  minimum: float | None
  maximum: float | None = None
  nullable: bool = False


_SPECS: Final = MappingProxyType(
  {
    "writer_soft_turn_budget": _Spec(integer=True, minimum=1, nullable=True),
    "child_soft_turn_budget": _Spec(integer=True, minimum=1),
    "fork_soft_turn_budget": _Spec(integer=True, minimum=1),
    "soft_budget_final_factor": _Spec(integer=False, minimum=1.0),
    "soft_budget_grace_turns": _Spec(integer=True, minimum=0),
    "child_silence_timeout_seconds": _Spec(integer=False, minimum=0.0),
    "max_spawn_depth": _Spec(integer=True, minimum=None),  # any integer: negative disables the bound
    "max_concurrent_children": _Spec(integer=True, minimum=1),
    "repeat_tool_error_limit": _Spec(integer=True, minimum=1, nullable=True),
    "model_context_limit": _Spec(integer=True, minimum=1),
    "context_warning_pct": _Spec(integer=True, minimum=1, maximum=100),
    "context_pressure_reminder_pct": _Spec(integer=True, minimum=1, maximum=100),
    "context_pressure_reminder_step_pct": _Spec(integer=True, minimum=1),
    "compaction_trigger_pct": _Spec(integer=True, minimum=1, maximum=100),
  }
)
_KEYS: Final = frozenset(_SPECS)


class RunSettingsError(ValueError):
  """The settings (packaged defaults or operator file) are unreadable or invalid."""


@dataclass(frozen=True)
class RunSettings:
  writer_soft_turn_budget: int | None
  child_soft_turn_budget: int
  fork_soft_turn_budget: int
  soft_budget_final_factor: float
  soft_budget_grace_turns: int
  child_silence_timeout_seconds: float
  max_spawn_depth: int
  max_concurrent_children: int
  repeat_tool_error_limit: int | None
  model_context_limit: int
  context_warning_pct: int
  context_pressure_reminder_pct: int
  context_pressure_reminder_step_pct: int
  compaction_trigger_pct: int

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  def soft_turn_threshold(self, kind: str) -> int | None:
    """One session kind's steer threshold in provider requests; ``None`` means no ladder.

    ``"writer"`` and ``"fork"`` have their own bounds; every other kind (``sub_agent``,
    ``autonomous``) is a child.
    """

    if kind == "writer":
      return self.writer_soft_turn_budget
    if kind == "fork":
      return self.fork_soft_turn_budget
    return self.child_soft_turn_budget


def _check(key: str, value: object) -> int | float | None:
  spec = _SPECS[key]
  if value is None:
    if spec.nullable:
      return None
    raise RunSettingsError(f"{key} must not be null")
  kind = "an integer" if spec.integer else "a number"
  if spec.integer:
    valid = isinstance(value, int) and not isinstance(value, bool)
  else:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
  if not valid or not isinstance(value, (int, float)):
    raise RunSettingsError(f"{key} must be {kind}{' or null' if spec.nullable else ''}, got {value!r}")
  if spec.minimum is not None and (value < spec.minimum or (spec.maximum is not None and value > spec.maximum)):
    bound = f">= {spec.minimum:g}" if spec.maximum is None else f"{spec.minimum:g}..{spec.maximum:g}"
    raise RunSettingsError(f"{key} must be {kind} {bound}, got {value!r}")
  return value if spec.integer else float(value)


def _parse(raw: object) -> RunSettings:
  if not isinstance(raw, dict):
    raise RunSettingsError("run settings must be a JSON object")
  unknown = set(raw) - _KEYS
  missing = _KEYS - set(raw)
  if unknown or missing:
    raise RunSettingsError(f"run settings keys: unknown {sorted(unknown)}, missing {sorted(missing)}")
  values: dict[str, Any] = {key: _check(key, raw[key]) for key in _SPECS}
  return RunSettings(**values)


_defaults_lock = threading.Lock()
_defaults_cache: dict[str, Any] | None = None


def _packaged_defaults() -> dict[str, Any]:
  global _defaults_cache
  with _defaults_lock:
    cached = _defaults_cache
    if cached is None:
      try:
        cached = json.loads(_DEFAULTS_PATH.read_text(encoding="utf-8"))
        _parse(cached)
      except (OSError, ValueError) as exc:
        raise RunSettingsError(f"packaged run defaults {_DEFAULTS_PATH} are invalid: {exc}") from exc
      _defaults_cache = cached
    return copy.deepcopy(cached)


def settings_path(environ: Mapping[str, str] | None = None) -> Path:
  return operator_settings.settings_path(SETTINGS_PATH_ENV, DEFAULT_SETTINGS_PATH, environ)


def settings_lock_path(environ: Mapping[str, str] | None = None) -> Path:
  """The sibling lock file every settings writer and (when it can open it) reader flocks."""

  return operator_settings.lock_path(settings_path(environ))


_cache: operator_settings.StampCache[RunSettings] = operator_settings.StampCache()


def _merged_settings(overrides: dict[str, Any], path: Path) -> RunSettings:
  defaults = _packaged_defaults()
  try:
    return _parse(operator_settings.merge(defaults, overrides))
  except RunSettingsError as exc:
    raise RunSettingsError(f"run settings file {path}: {exc}") from exc


def load_run_settings() -> RunSettings:
  """Packaged defaults deep-merged under the operator file; re-stat'ed (mtime_ns + size) every call."""

  path = settings_path()
  return _cache.load(path, lambda raw: _merged_settings(raw, path), error=RunSettingsError, label=_LABEL)


def pinned_run_settings(**overrides: object) -> RunSettings:
  """Packaged defaults plus ``overrides``; never reads the operator file."""

  return _parse(operator_settings.merge(_packaged_defaults(), overrides))


def operator_overrides() -> dict[str, object]:
  """The raw operator file ({} if absent); a malformed file raises ``RunSettingsError``."""

  return operator_settings.read_operator_file(settings_path(), error=RunSettingsError, label=_LABEL)[0]


def _cli_value(text: str) -> object:
  if text.strip().lower() in ("null", "none"):
    return None
  try:
    return json.loads(text)
  except ValueError:
    return text


def set_operator_setting(key: str, value: object) -> RunSettings:
  """Set one bound in the operator file and return the merged result.

  A ``str`` value is the CLI form: ``"null"`` / ``"none"`` write ``null`` (an override
  that clears a nullable bound; ``null`` on a bound that forbids it is rejected), anything
  else is parsed as a JSON scalar, and text that is not JSON stays text and fails
  validation. The merged result is validated before the in-place write (the file keeps
  its inode); invalid input leaves the file's bytes unchanged.
  """

  if key not in _KEYS:
    raise RunSettingsError(f"unknown run setting {key!r}; known {sorted(_KEYS)}")
  if isinstance(value, str):
    value = _cli_value(value)
  path = settings_path()
  with operator_settings.settings_lock(path):
    raw, _ = operator_settings.read_operator_file(path, error=RunSettingsError, label=_LABEL, locked=True)
    overrides = copy.deepcopy(raw)
    overrides[key] = value
    _merged_settings(overrides, path)  # validate before anything is written
    operator_settings.write_operator_file_in_place(path, overrides)
  return load_run_settings()


__all__ = [
  "DEFAULT_SETTINGS_PATH",
  "RunSettings",
  "RunSettingsError",
  "SETTINGS_PATH_ENV",
  "load_run_settings",
  "operator_overrides",
  "pinned_run_settings",
  "set_operator_setting",
  "settings_lock_path",
  "settings_path",
]
