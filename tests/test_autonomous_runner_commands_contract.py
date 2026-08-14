from __future__ import annotations

import pytest

from agent_gateway.autonomous_runner_commands import build_autonomous_cmd


def _build(**overrides):
  arguments = {
    "python_executable": "/usr/bin/python3",
    "profile": "research_producer",
    "mode": "once",
  }
  arguments.update(overrides)
  return build_autonomous_cmd(**arguments)


def test_skill_command_carries_budget_and_delivery_suppression() -> None:
  assert _build(
    mode="skill",
    skill="thesis-review",
    ticker="foo",
    context="context",
    max_budget_usd=12,
    deliver=False,
  ) == [
    "/usr/bin/python3",
    "-m",
    "agent.autonomous",
    "--profile",
    "research_producer",
    "--skill",
    "thesis-review",
    "--no-deliver",
    "--max-budget-usd",
    "12.0",
    "--ticker",
    "FOO",
    "--context",
    "context",
  ]


def test_pack_command_is_closed_to_exact_pack_name() -> None:
  assert _build(mode="pack", pack=" morning-pack ") == [
    "/usr/bin/python3",
    "-m",
    "agent.autonomous",
    "--profile",
    "research_producer",
    "--pack",
    "morning-pack",
  ]

  with pytest.raises(ValueError, match="only accepts the pack"):
    _build(mode="pack", pack="morning-pack", ticker="FOO")
  with pytest.raises(ValueError, match="deliver=False requires"):
    _build(mode="pack", pack="morning-pack", deliver=False)


@pytest.mark.parametrize(
  "overrides",
  [
    {"mode": "once", "ticker": "FOO"},
    {"mode": "task", "task": "work", "ticker": "FOO"},
    {"mode": "skill", "skill": "thesis-review", "pack": "morning"},
    {"mode": "skill", "skill": "thesis-review", "deliver": 0},
  ],
)
def test_closed_command_contract_rejects_cross_mode_fields(overrides) -> None:
  with pytest.raises(ValueError):
    _build(**overrides)
