"""Run-and-loop diagnostic bounds: packaged defaults under one live operator file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_gateway import run_settings
from agent_gateway.run_settings import (
  RunSettingsError,
  load_run_settings,
  operator_overrides,
  pinned_run_settings,
  set_operator_setting,
  settings_lock_path,
  settings_path,
)

PACKAGED = json.loads(Path(run_settings.__file__).with_name("run_settings.defaults.json").read_text())


@pytest.fixture(autouse=True)
def operator_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  path = tmp_path / "config" / "run-settings.json"
  monkeypatch.setenv("HANK_RUN_SETTINGS_PATH", str(path))
  run_settings._cache.clear()
  return path


def test_without_an_operator_file_the_packaged_defaults_apply(operator_file: Path) -> None:
  assert not operator_file.exists()
  assert settings_path() == operator_file
  assert load_run_settings().to_dict() == PACKAGED
  assert len(PACKAGED) == 14
  assert operator_overrides() == {}


def test_set_is_seen_by_the_next_load_and_written_in_place(operator_file: Path) -> None:
  assert load_run_settings().writer_soft_turn_budget is None

  assert set_operator_setting("writer_soft_turn_budget", 30).writer_soft_turn_budget == 30
  inode = operator_file.stat().st_ino
  set_operator_setting("context_warning_pct", 90)

  assert operator_file.stat().st_ino == inode
  assert operator_overrides() == {"writer_soft_turn_budget": 30, "context_warning_pct": 90}
  assert settings_lock_path() == operator_file.with_name("run-settings.json.lock")
  assert load_run_settings().context_warning_pct == 90


def test_a_hand_edited_file_is_seen_without_a_restart(operator_file: Path) -> None:
  operator_file.parent.mkdir(parents=True)
  operator_file.write_text('{"fork_soft_turn_budget": 4}\n')
  assert load_run_settings().fork_soft_turn_budget == 4

  operator_file.write_text('{"fork_soft_turn_budget": 7}\n')
  stat = operator_file.stat()
  os.utime(operator_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

  assert load_run_settings().fork_soft_turn_budget == 7


def test_the_cli_text_form_parses_json_and_null(operator_file: Path) -> None:
  assert set_operator_setting("writer_soft_turn_budget", "40").writer_soft_turn_budget == 40
  assert set_operator_setting("soft_budget_final_factor", "2.5").soft_budget_final_factor == 2.5

  assert set_operator_setting("writer_soft_turn_budget", "null").writer_soft_turn_budget is None
  assert operator_overrides()["writer_soft_turn_budget"] is None
  set_operator_setting("repeat_tool_error_limit", "3")
  assert set_operator_setting("repeat_tool_error_limit", "none").repeat_tool_error_limit is None


@pytest.mark.parametrize(
  ("key", "value"),
  [
    ("writer_soft_turn_budget", 0),
    ("child_soft_turn_budget", "many"),
    ("child_soft_turn_budget", 1.5),
    ("child_soft_turn_budget", True),
    ("child_soft_turn_budget", None),
    ("soft_budget_final_factor", 0.9),
    ("soft_budget_grace_turns", -1),
    ("child_silence_timeout_seconds", -1),
    ("max_spawn_depth", 1.5),
    ("max_concurrent_children", 0),
    ("context_warning_pct", 101),
    ("compaction_trigger_pct", 0),
    ("model_context_limit", 0),
    ("child_soft_turn_budget", {"nested": 1}),
    ("not_a_setting", 1),
    ("context_warning_pct.nested", 1),
  ],
)
def test_a_rejected_set_leaves_the_file_bytes_unchanged(operator_file: Path, key: str, value: object) -> None:
  set_operator_setting("fork_soft_turn_budget", 5)
  before = operator_file.read_bytes()

  with pytest.raises(RunSettingsError):
    set_operator_setting(key, value)

  assert operator_file.read_bytes() == before
  assert load_run_settings().fork_soft_turn_budget == 5


def test_a_negative_spawn_depth_is_accepted_as_the_disabled_bound(operator_file: Path) -> None:
  assert set_operator_setting("max_spawn_depth", -1).max_spawn_depth == -1
  assert pinned_run_settings(max_spawn_depth=-1).max_spawn_depth == -1


def test_a_rejected_set_creates_no_file(operator_file: Path) -> None:
  with pytest.raises(RunSettingsError):
    set_operator_setting("max_spawn_depth", "deep")

  assert not operator_file.exists()


def test_pinned_ignores_the_operator_file(operator_file: Path) -> None:
  set_operator_setting("writer_soft_turn_budget", 9)
  operator_file.write_text("not json at all")

  pinned = pinned_run_settings(max_spawn_depth=1)

  assert pinned.max_spawn_depth == 1
  assert pinned.writer_soft_turn_budget is None
  assert {**pinned.to_dict(), "max_spawn_depth": 2} == PACKAGED
  with pytest.raises(RunSettingsError):
    pinned_run_settings(max_concurrent_children=0)
  with pytest.raises(RunSettingsError):
    pinned_run_settings(unknown_bound=1)


def test_soft_turn_threshold_by_kind(operator_file: Path) -> None:
  settings = load_run_settings()
  assert settings.soft_turn_threshold("writer") is None
  assert settings.soft_turn_threshold("fork") == 10
  assert settings.soft_turn_threshold("sub_agent") == 100
  assert settings.soft_turn_threshold("autonomous") == 100

  set_operator_setting("writer_soft_turn_budget", 25)
  set_operator_setting("child_soft_turn_budget", 50)
  set_operator_setting("fork_soft_turn_budget", 3)
  settings = load_run_settings()

  assert settings.soft_turn_threshold("writer") == 25
  assert settings.soft_turn_threshold("fork") == 3
  assert settings.soft_turn_threshold("sub_agent") == 50
  assert settings.soft_turn_threshold("autonomous") == 50
  assert pinned_run_settings(writer_soft_turn_budget=2).soft_turn_threshold("writer") == 2


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"unknown_bound": 1}', '{"max_spawn_depth": "deep"}'])
def test_a_bad_operator_file_raises_at_the_next_read(operator_file: Path, content: str) -> None:
  operator_file.parent.mkdir(parents=True)
  operator_file.write_text(content)

  with pytest.raises(RunSettingsError):
    load_run_settings()
