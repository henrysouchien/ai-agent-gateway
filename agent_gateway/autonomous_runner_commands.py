from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .fixture_gate import is_fixture_profile_name, is_fixture_skill_name, require_fixture_provider_available
from .artifact_paths import canonicalize_ticker

_AUTONOMOUS_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def normalize_autonomous_profile(
  profile: str,
  *,
  is_fixture_profile_name_func: Callable[[str], bool] = is_fixture_profile_name,
  profile_name_re: Any = _AUTONOMOUS_PROFILE_NAME_RE,
) -> str:
  normalized_profile = str(profile or "").strip().lower()
  if not normalized_profile:
    raise ValueError("profile is required")
  if is_fixture_profile_name_func(normalized_profile):
    return normalized_profile
  if not profile_name_re.fullmatch(normalized_profile):
    raise ValueError("profile must be a Python module-safe name using letters, numbers, and underscores")
  return normalized_profile


def build_autonomous_cmd(
  *,
  python_executable: str,
  profile: str,
  mode: str,
  task: str | None,
  skill: str | None,
  context: str | None,
  ticker: str | None = None,
  dev_mode: bool = False,
  normalize_autonomous_profile_func: Callable[[str], str] = normalize_autonomous_profile,
  is_fixture_profile_name_func: Callable[[str], bool] = is_fixture_profile_name,
  is_fixture_skill_name_func: Callable[[str], bool] = is_fixture_skill_name,
  require_fixture_provider_available_func: Callable[..., None] = require_fixture_provider_available,
) -> list[str]:
  normalized_profile = normalize_autonomous_profile_func(profile)
  if is_fixture_profile_name_func(normalized_profile):
    require_fixture_provider_available_func("fixture profile dispatch", error_type=ValueError)

  normalized_mode = mode.strip().lower()
  if normalized_mode not in {"once", "task", "skill"}:
    raise ValueError("mode must be once, task, or skill")

  if dev_mode and normalized_mode == "task":
    raise ValueError("dev_mode is implicit for mode='task'; do not pass dev_mode=True")
  if dev_mode and normalized_mode == "once":
    raise ValueError("dev_mode requires mode='skill'; use mode='task' for dev tasks instead")

  cmd = [python_executable, "-m", "agent.autonomous", "--profile", normalized_profile]
  if dev_mode:
    cmd.append("--dev")

  if normalized_mode == "once":
    if task or skill or context:
      raise ValueError("mode='once' does not accept task, skill, or context")
    return cmd

  if normalized_mode == "task":
    if not task or not task.strip():
      raise ValueError("task is required when mode='task'")
    if skill or context:
      raise ValueError("mode='task' only accepts the task parameter")
    cmd.extend(["--task", task.strip()])
    return cmd

  if not skill or not skill.strip():
    raise ValueError("skill is required when mode='skill'")
  if is_fixture_skill_name_func(skill):
    require_fixture_provider_available_func("fixture skill dispatch", error_type=ValueError)
  if task:
    raise ValueError("mode='skill' does not accept task")
  cmd.extend(["--skill", skill.strip()])
  if ticker and ticker.strip():
    cmd.extend(["--ticker", canonicalize_ticker(ticker)])
  if context and context.strip():
    cmd.extend(["--context", context.strip()])
  return cmd


__all__ = [
  "_AUTONOMOUS_PROFILE_NAME_RE",
  "build_autonomous_cmd",
  "normalize_autonomous_profile",
]
