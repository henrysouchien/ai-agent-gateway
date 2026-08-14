from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.canonical_json_target_lock as target_lock_module  # noqa: E402
from agent_gateway.canonical_json_target_lock import (  # noqa: E402
  CanonicalJsonTargetLockError,
  CanonicalJsonTargetLockTimeout,
  MAX_CANONICAL_JSON_TARGET_BYTES,
  canonical_json_target_lock_name,
  lock_canonical_json_path,
  write_locked_canonical_json_target,
)
from agent_gateway.skill_completion_wal import (  # noqa: E402
  SkillCompletionEffectConflict,
  TopLevelSkillCompletionEffectPlan,
  apply_completion_effect,
)
from agent_gateway.skills import SkillStateStore  # noqa: E402


def _hold_target_lock(
  target_path: str,
  acquired: Any,
  release: Any,
) -> None:
  with lock_canonical_json_path(Path(target_path)):
    acquired.set()
    if not release.wait(10):
      raise TimeoutError("test did not release held target lock")


def _increment_skill_state(
  target_path: str,
  start: Any,
  mutation_barrier: Any,
) -> None:
  if not start.wait(10):
    raise TimeoutError("test did not release increment worker")
  store = SkillStateStore(Path(target_path))

  def _increment(previous: dict[str, Any]) -> dict[str, Any]:
    try:
      mutation_barrier.wait(timeout=0.5)
    except threading.BrokenBarrierError:
      pass
    return {
      **previous,
      "run_count": int(previous.get("run_count", 0) or 0) + 1,
    }

  store.update("skill-a", _increment)


@pytest.mark.parametrize(
  "attack",
  ("symlink", "hardlink", "mode"),
)
def test_lock_file_attacks_fail_closed(
  tmp_path: Path,
  attack: str,
) -> None:
  target_path = tmp_path / "skill_state.json"
  lock_path = tmp_path / canonical_json_target_lock_name(
    target_path.name
  )
  if attack == "symlink":
    decoy = tmp_path / "decoy"
    decoy.write_bytes(b"")
    lock_path.symlink_to(decoy)
  elif attack == "hardlink":
    decoy = tmp_path / "decoy"
    decoy.write_bytes(b"")
    decoy.chmod(0o600)
    os.link(decoy, lock_path)
  else:
    lock_path.write_bytes(b"")
    lock_path.chmod(0o644)

  with pytest.raises(CanonicalJsonTargetLockError):
    SkillStateStore(target_path).set(
      "skill-a",
      {"run_count": 1},
    )

  assert not target_path.exists()


@pytest.mark.parametrize(
  "attack",
  ("symlink", "hardlink", "mode"),
)
def test_target_file_attacks_fail_closed(
  tmp_path: Path,
  attack: str,
) -> None:
  target_path = tmp_path / "skill_state.json"
  decoy = tmp_path / "decoy.json"
  decoy.write_text('{"sentinel":true}\n', encoding="utf-8")
  decoy.chmod(0o600)
  if attack == "symlink":
    target_path.symlink_to(decoy)
  elif attack == "hardlink":
    os.link(decoy, target_path)
  else:
    target_path.write_text("{}", encoding="utf-8")
    target_path.chmod(0o666)

  with pytest.raises(CanonicalJsonTargetLockError):
    SkillStateStore(target_path).set(
      "skill-a",
      {"run_count": 1},
    )

  assert json.loads(decoy.read_text(encoding="utf-8")) == {
    "sentinel": True,
  }


def test_lock_binding_swap_is_detected_when_body_also_fails(
  tmp_path: Path,
) -> None:
  target_path = tmp_path / "skill_state.json"

  with pytest.raises(
    CanonicalJsonTargetLockError,
    match="binding changed",
  ):
    with lock_canonical_json_path(target_path) as target:
      lock_path = tmp_path / target.lock_name
      displaced = tmp_path / "displaced.lock"
      lock_path.rename(displaced)
      lock_path.write_bytes(b"")
      lock_path.chmod(0o600)
      raise RuntimeError("protected body failed")


def test_lock_is_released_when_mutation_raises(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  target_path = tmp_path / "skill_state.json"
  store = SkillStateStore(target_path)

  def _fail(_payload: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("injected mutation failure")

  with pytest.raises(RuntimeError, match="injected"):
    store.mutate(_fail)

  monkeypatch.setattr(
    target_lock_module,
    "CANONICAL_JSON_TARGET_LOCK_TIMEOUT_SECONDS",
    0.05,
  )
  store.set("skill-a", {"run_count": 1})
  assert store.get("skill-a") == {"run_count": 1}


def test_path_lock_securely_creates_missing_parent_components(
  tmp_path: Path,
) -> None:
  target_path = (
    tmp_path
    / "workspace"
    / "notes"
    / "analyst"
    / "skill_state.json"
  )

  SkillStateStore(target_path).set(
    "skill-a",
    {"run_count": 1},
  )

  assert SkillStateStore(target_path).get("skill-a") == {
    "run_count": 1,
  }


def test_path_lock_rejects_intermediate_symlink(
  tmp_path: Path,
) -> None:
  external = tmp_path / "external"
  (external / "analyst").mkdir(parents=True)
  notes_link = tmp_path / "workspace" / "notes"
  notes_link.parent.mkdir(parents=True)
  notes_link.symlink_to(external)
  target_path = notes_link / "analyst" / "skill_state.json"

  with pytest.raises(CanonicalJsonTargetLockError):
    SkillStateStore(target_path).set(
      "skill-a",
      {"run_count": 1},
    )

  assert not (external / "analyst" / "skill_state.json").exists()


def test_maximum_canonical_body_allows_trailing_newline(
  tmp_path: Path,
) -> None:
  target_path = tmp_path / "skill_state.json"
  empty_payload_size = len(b'{"value":""}')
  value = "x" * (
    MAX_CANONICAL_JSON_TARGET_BYTES - empty_payload_size
  )

  with lock_canonical_json_path(target_path) as target:
    snapshot = write_locked_canonical_json_target(
      target,
      {"value": value},
    )

  assert snapshot.raw is not None
  assert len(snapshot.raw) == MAX_CANONICAL_JSON_TARGET_BYTES + 1


def test_lock_wait_is_bounded_and_released_by_holder(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  target_path = tmp_path / "skill_state.json"
  context = multiprocessing.get_context("spawn")
  acquired = context.Event()
  release = context.Event()
  holder = context.Process(
    target=_hold_target_lock,
    args=(str(target_path), acquired, release),
  )
  holder.start()
  try:
    assert acquired.wait(10)
    monkeypatch.setattr(
      target_lock_module,
      "CANONICAL_JSON_TARGET_LOCK_TIMEOUT_SECONDS",
      0.05,
    )
    with pytest.raises(CanonicalJsonTargetLockTimeout):
      SkillStateStore(target_path).set(
        "skill-a",
        {"run_count": 1},
      )
  finally:
    release.set()
    holder.join(10)
    if holder.is_alive():
      holder.terminate()
      holder.join(10)
  assert holder.exitcode == 0

  SkillStateStore(target_path).set(
    "skill-a",
    {"run_count": 1},
  )


def test_multiprocess_skill_state_updates_preserve_both_increments(
  tmp_path: Path,
) -> None:
  target_path = tmp_path / "skill_state.json"
  context = multiprocessing.get_context("spawn")
  start = context.Event()
  mutation_barrier = context.Barrier(2)
  workers = [
    context.Process(
      target=_increment_skill_state,
      args=(str(target_path), start, mutation_barrier),
    )
    for _index in range(2)
  ]
  for worker in workers:
    worker.start()
  start.set()
  for worker in workers:
    worker.join(10)
    if worker.is_alive():
      worker.terminate()
      worker.join(10)
  assert [worker.exitcode for worker in workers] == [0, 0]
  assert SkillStateStore(target_path).get("skill-a") == {
    "run_count": 2,
  }


def test_same_before_lifecycle_effects_cannot_both_report_applied(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "workspace"
  target_path = workspace / "notes" / "skill_state.json"
  target_path.parent.mkdir(parents=True)
  target_path.write_text(
    json.dumps({"skill-a": {"run_count": 1}}),
    encoding="utf-8",
  )

  def _next_state(
    _exists: bool,
    before: Any | None,
  ) -> dict[str, Any]:
    payload = dict(before) if isinstance(before, dict) else {}
    payload["skill-a"] = {"run_count": 2}
    return payload

  plans = [
    TopLevelSkillCompletionEffectPlan.canonical_json_update(
      workspace_path=workspace,
      target_path=target_path,
      update=_next_state,
    )
    for _index in range(2)
  ]
  start = threading.Barrier(3)
  outcomes: list[str] = []
  errors: list[BaseException] = []

  def _apply(
    plan: TopLevelSkillCompletionEffectPlan,
  ) -> None:
    try:
      start.wait()
      outcomes.append(
        apply_completion_effect(
          plan.durable_payload(),
          expected_workspace=workspace,
        )
      )
    except BaseException as exc:
      errors.append(exc)

  threads = [
    threading.Thread(target=_apply, args=(plan,))
    for plan in plans
  ]
  for thread in threads:
    thread.start()
  start.wait()
  for thread in threads:
    thread.join(10)

  assert not errors
  assert not any(thread.is_alive() for thread in threads)
  assert sorted(outcomes) == ["already_applied", "applied"]


def test_different_same_before_lifecycle_effects_have_one_winner(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "workspace"
  target_path = workspace / "notes" / "skill_state.json"
  target_path.parent.mkdir(parents=True)
  target_path.write_text(
    json.dumps({"skill-a": {"run_count": 1}}),
    encoding="utf-8",
  )
  plans = [
    TopLevelSkillCompletionEffectPlan.canonical_json_update(
      workspace_path=workspace,
      target_path=target_path,
      update=lambda _exists, _before, source=source: {
        "skill-a": {
          "run_count": 2,
          "source": source,
        }
      },
    )
    for source in ("first", "second")
  ]
  start = threading.Barrier(3)
  outcomes: list[str] = []
  errors: list[BaseException] = []

  def _apply(
    plan: TopLevelSkillCompletionEffectPlan,
  ) -> None:
    try:
      start.wait()
      outcomes.append(
        apply_completion_effect(
          plan.durable_payload(),
          expected_workspace=workspace,
        )
      )
    except BaseException as exc:
      errors.append(exc)

  threads = [
    threading.Thread(target=_apply, args=(plan,))
    for plan in plans
  ]
  for thread in threads:
    thread.start()
  start.wait()
  for thread in threads:
    thread.join(10)

  assert not any(thread.is_alive() for thread in threads)
  assert outcomes == ["applied"]
  assert len(errors) == 1
  assert isinstance(errors[0], SkillCompletionEffectConflict)


def test_lifecycle_plan_conflicts_on_intervening_state_update(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "workspace"
  target_path = workspace / "notes" / "skill_state.json"
  target_path.parent.mkdir(parents=True)
  SkillStateStore(target_path).set(
    "skill-a",
    {"run_count": 1},
  )

  plan = TopLevelSkillCompletionEffectPlan.canonical_json_update(
    workspace_path=workspace,
    target_path=target_path,
    update=lambda _exists, before: {
      **(dict(before) if isinstance(before, dict) else {}),
      "skill-a": {"run_count": 2, "source": "lifecycle"},
    },
  )
  SkillStateStore(target_path).update(
    "skill-a",
    lambda previous: {
      **previous,
      "run_count": int(previous.get("run_count", 0)) + 1,
      "source": "peer",
    },
  )

  with pytest.raises(
    SkillCompletionEffectConflict,
    match="matches neither before nor after digest",
  ):
    apply_completion_effect(
      plan.durable_payload(),
      expected_workspace=workspace,
    )
  assert SkillStateStore(target_path).get("skill-a") == {
    "run_count": 2,
    "source": "peer",
  }
