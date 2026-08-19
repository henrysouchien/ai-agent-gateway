from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from .artifact_paths import canonicalize_ticker

_AUTONOMOUS_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def normalize_max_budget_usd(value: float | None) -> float | None:
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError("max_budget_usd must be a finite positive number")
  normalized = float(value)
  if not math.isfinite(normalized) or normalized <= 0:
    raise ValueError("max_budget_usd must be a finite positive number")
  return normalized


def normalize_autonomous_profile(
  profile: str,
  *,
  profile_name_re: Any = _AUTONOMOUS_PROFILE_NAME_RE,
) -> str:
  normalized_profile = str(profile or "").strip().lower()
  if not normalized_profile:
    raise ValueError("profile is required")
  if not profile_name_re.fullmatch(normalized_profile):
    raise ValueError("profile must be a Python module-safe name using letters, numbers, and underscores")
  return normalized_profile


def build_autonomous_cmd(
  *,
  python_executable: str,
  profile: str,
  mode: str,
  task: str | None = None,
  skill: str | None = None,
  pack: str | None = None,
  deliver: bool = True,
  context: str | None = None,
  ticker: str | None = None,
  max_budget_usd: float | None = None,
  normalize_autonomous_profile_func: Callable[[str], str] = normalize_autonomous_profile,
) -> list[str]:
  normalized_profile = normalize_autonomous_profile_func(profile)

  normalized_mode = mode.strip().lower()
  if normalized_mode not in {"once", "task", "skill", "pack"}:
    raise ValueError("mode must be once, task, skill, or pack")
  if type(deliver) is not bool:
    raise ValueError("deliver must be a boolean")
  normalized_max_budget_usd = normalize_max_budget_usd(max_budget_usd)
  if normalized_max_budget_usd is not None and normalized_mode != "skill":
    raise ValueError("max_budget_usd requires mode='skill'")
  if not deliver and normalized_mode != "skill":
    raise ValueError("deliver=False requires mode='skill'")

  cmd = [python_executable, "-m", "agent.autonomous", "--profile", normalized_profile]

  if normalized_mode == "once":
    if task or skill or pack or context or ticker:
      raise ValueError("mode='once' does not accept task, skill, pack, context, or ticker")
    return cmd

  if normalized_mode == "task":
    if not task or not task.strip():
      raise ValueError("task is required when mode='task'")
    if skill or pack or context or ticker:
      raise ValueError("mode='task' only accepts the task parameter")
    cmd.extend(["--task", task.strip()])
    return cmd

  if normalized_mode == "pack":
    if not pack or not pack.strip():
      raise ValueError("pack is required when mode='pack'")
    if task or skill or context or ticker:
      raise ValueError("mode='pack' only accepts the pack parameter")
    cmd.extend(["--pack", pack.strip()])
    return cmd

  if not skill or not skill.strip():
    raise ValueError("skill is required when mode='skill'")
  if task or pack:
    raise ValueError("mode='skill' does not accept task or pack")
  cmd.extend(["--skill", skill.strip()])
  if not deliver:
    cmd.append("--no-deliver")
  if normalized_max_budget_usd is not None:
    cmd.extend(["--max-budget-usd", str(normalized_max_budget_usd)])
  if ticker and ticker.strip():
    cmd.extend(["--ticker", canonicalize_ticker(ticker)])
  if context and context.strip():
    cmd.extend(["--context", context.strip()])
  return cmd


__all__ = [
  "_AUTONOMOUS_PROFILE_NAME_RE",
  "build_autonomous_cmd",
  "normalize_max_budget_usd",
  "normalize_autonomous_profile",
]
