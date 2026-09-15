from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_gateway.skill_limits import (  # noqa: E402
  ActiveSkillAdmission,
  SkillExecutionLimits,
)
from agent_gateway.skill_context import (  # noqa: E402
  clear_current_skill,
  current_skill,
  current_skill_admission,
  reset_current_skill,
  set_current_skill,
)


def _admission(name: str = "quant-research") -> ActiveSkillAdmission:
  return ActiveSkillAdmission(
    name,
    SkillExecutionLimits(20, 32_000, 20.0),
  )


def test_current_skill_is_only_name_projection_of_exact_pair() -> None:
  clear_current_skill()
  admission = _admission()
  token = set_current_skill(admission)
  try:
    assert current_skill_admission() is admission
    assert current_skill() == "quant-research"
  finally:
    reset_current_skill(token)
  assert current_skill_admission() is None
  assert current_skill() is None


def test_set_current_skill_refuses_name_only_state() -> None:
  with pytest.raises(TypeError):
    set_current_skill("quant-research")  # type: ignore[arg-type]


def test_context_is_isolated_across_tasks() -> None:
  clear_current_skill()

  async def observe(name: str) -> tuple[str | None, str | None]:
    token = set_current_skill(_admission(name))
    try:
      await asyncio.sleep(0)
      return current_skill(), current_skill_admission().skill_name  # type: ignore[union-attr]
    finally:
      reset_current_skill(token)

  async def run() -> list[tuple[str | None, str | None]]:
    return list(await asyncio.gather(observe("one"), observe("two")))

  assert asyncio.run(run()) == [("one", "one"), ("two", "two")]
  assert current_skill() is None
