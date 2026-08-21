from __future__ import annotations

import builtins
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest

from agent_gateway.control_plane import batches as batches_module
from agent_gateway.control_plane import skills as skills_module
from agent_gateway.control_plane import valuation_ready_tools as valuation_ready_tools_module


ROOT = Path(__file__).resolve().parents[4]
BROAD_API_FALLBACK_RE = re.compile(r"except ModuleNotFoundError:\s*\n\s+from api\.")


@pytest.mark.parametrize(
  ("call_factory", "primary_name", "fallback_name"),
  [
    (lambda: skills_module._loader_api, "agent.skills", "api.agent.skills"),
    (
      lambda: valuation_ready_tools_module._valuation_ready_defaults,
      "agent.skills.diligence_tracks",
      "api.agent.skills.diligence_tracks",
    ),
    (lambda: batches_module._controller, "agent.batch", "api.agent.batch"),
    (
      lambda: batches_module._batch_workflow_catalog,
      "agent.skills.diligence_tracks",
      "api.agent.skills.diligence_tracks",
    ),
    (lambda: batches_module._active_batch_error_type, "agent.batch.registry", "api.agent.batch.registry"),
  ],
)
def test_control_plane_fallbacks_raise_when_primary_dependency_breaks(
  monkeypatch: pytest.MonkeyPatch,
  call_factory: Callable[[], Callable[[], Any]],
  primary_name: str,
  fallback_name: str,
) -> None:
  calls: list[str] = []
  real_import = builtins.__import__

  def fake_import(
    name: str,
    globals_: Any = None,
    locals_: Any = None,
    fromlist: Any = (),
    level: int = 0,
  ) -> Any:
    if name == primary_name:
      calls.append(name)
      raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")
    if name == fallback_name:
      calls.append(name)
      raise AssertionError(f"{fallback_name} fallback should not run for dependency drift")
    return real_import(name, globals_, locals_, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", fake_import)

  with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
    call_factory()()
  assert calls == [primary_name]


def test_registry_fallback_raises_when_batch_registry_dependency_breaks(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  real_import = builtins.__import__
  memory_module = ModuleType("memory")
  memory_module.get_workspace_dir = lambda _user_id: str(tmp_path)  # type: ignore[attr-defined]
  calls: list[str] = []

  def fake_import(
    name: str,
    globals_: Any = None,
    locals_: Any = None,
    fromlist: Any = (),
    level: int = 0,
  ) -> Any:
    if name == "memory":
      return memory_module
    if name == "agent.batch.registry":
      calls.append(name)
      raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")
    if name in {"api.memory", "api.agent.batch.registry"}:
      calls.append(name)
      raise AssertionError(f"{name} fallback should not run for dependency drift")
    return real_import(name, globals_, locals_, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", fake_import)

  with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
    batches_module._registry_for_user("alice")
  assert calls == ["agent.batch.registry"]


def test_control_plane_api_fallbacks_check_missing_module_name() -> None:
  offenders = [
    file_path.relative_to(ROOT).as_posix()
    for file_path in (
      ROOT / "packages" / "agent-gateway" / "agent_gateway" / "control_plane"
    ).rglob("*.py")
    if BROAD_API_FALLBACK_RE.search(file_path.read_text(encoding="utf-8"))
  ]

  assert offenders == []
