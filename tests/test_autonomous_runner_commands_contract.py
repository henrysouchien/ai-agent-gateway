from __future__ import annotations

import pytest

from agent_gateway.autonomous_runner_commands import build_autonomous_cmd


def test_skill_context_projection_defers_to_os_argument_budget(monkeypatch) -> None:
  from agent_gateway import autonomous_runner_commands

  monkeypatch.setattr(
    autonomous_runner_commands.os,
    "sysconf",
    lambda name: {"SC_PAGE_SIZE": 4096, "SC_ARG_MAX": 1}[name],
  )
  cmd = _build(mode="skill", skill="earnings-review", context="Ticker: MSFT")
  assert "--context" not in cmd


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
  ]


def test_direct_skill_command_carries_context_projection() -> None:
  cmd = _build(
    mode="skill",
    skill="earnings-review",
    context=" Ticker: MSFT ",
  )
  assert cmd[cmd.index("--context") + 1] == "Ticker: MSFT"


def test_direct_skill_large_context_remains_envelope_only(monkeypatch) -> None:
  from agent_gateway import autonomous_runner_commands

  monkeypatch.setattr(
    autonomous_runner_commands.os,
    "sysconf",
    lambda name: {"SC_PAGE_SIZE": 4096, "SC_ARG_MAX": 2 * 1024 * 1024}[name],
  )
  monkeypatch.setattr(autonomous_runner_commands.os, "environ", {})
  cmd = _build(
    mode="skill",
    skill="earnings-review",
    context="x" * (200 * 1024),
  )
  assert "--context" not in cmd


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
    {"mode": "once", "context": "review"},
    {"mode": "task", "task": "work", "context": "review"},
    {"mode": "pack", "pack": "morning", "context": "review"},
    {"mode": "skill", "skill": "thesis-review", "pack": "morning"},
    {"mode": "skill", "skill": "thesis-review", "deliver": 0},
  ],
)
def test_closed_command_contract_rejects_cross_mode_fields(overrides) -> None:
  with pytest.raises(ValueError):
    _build(**overrides)
