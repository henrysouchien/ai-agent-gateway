from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from agent_gateway.control_plane import profiles as profiles_module


ROOT = Path(__file__).resolve().parents[3]
BROAD_MODULE_NOT_FOUND_RE = re.compile(r"except ModuleNotFoundError:\s*\n")


def test_control_profiles_fallback_raises_when_primary_dependency_breaks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[str] = []

  def fake_import_module(name: str) -> Any:
    if name == "agent.profiles":
      calls.append(name)
      raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")
    if name == "api.agent.profiles":
      calls.append(name)
      raise AssertionError("api.agent.profiles fallback should not run for dependency drift")
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(profiles_module.importlib, "import_module", fake_import_module)

  with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
    profiles_module._profiles_api()
  assert calls == ["agent.profiles"]


def test_gateway_package_module_not_found_fallbacks_are_guarded() -> None:
  offenders = [
    file_path.relative_to(ROOT).as_posix()
    for file_path in (ROOT / "packages" / "agent-gateway" / "agent_gateway").rglob("*.py")
    if BROAD_MODULE_NOT_FOUND_RE.search(file_path.read_text(encoding="utf-8"))
  ]

  assert offenders == []
