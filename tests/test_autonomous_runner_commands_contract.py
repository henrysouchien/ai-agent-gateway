from __future__ import annotations

import pytest

from agent_gateway.autonomous_runner_commands import build_autonomous_cmd


def _build(**overrides):
  overrides.setdefault("python_executable", "/usr/bin/python3")
  overrides.setdefault("profile", "research_producer")
  overrides.setdefault("mode", "once")
  return build_autonomous_cmd(**overrides)


def test_skill_command_carries_delivery_suppression() -> None:
  assert _build(
    mode="skill",
    skill="thesis-review",
    ticker="foo",
    context="context",
    deliver=False,
  ) == [
    "/usr/bin/python3",
    "-m",
    "agent.autonomous",
    "--profile",
    "research_producer",
    "--no-deliver",
    "--skill",
    "thesis-review",
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


@pytest.mark.parametrize(
  ("overrides", "tail"),
  [
    ({"mode": "once"}, []),
    ({"mode": "task", "task": "work"}, ["--task", "work"]),
    ({"mode": "pack", "pack": "morning-pack"}, ["--pack", "morning-pack"]),
  ],
)
def test_every_mode_carries_delivery_suppression(overrides, tail) -> None:
  assert _build(deliver=False, **overrides) == [
    "/usr/bin/python3",
    "-m",
    "agent.autonomous",
    "--profile",
    "research_producer",
    "--no-deliver",
    *tail,
  ]


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
