from __future__ import annotations

import builtins
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from agent_gateway import autonomous as autonomous_module
from agent_gateway import autonomous_excel_dispatch as excel_dispatch_module
from agent_gateway.control_plane import profiles as profiles_module


ROOT = Path(__file__).resolve().parents[3]
BROAD_MODULE_NOT_FOUND_RE = re.compile(r"except ModuleNotFoundError:\s*\n")


@pytest.mark.parametrize(
  ("call", "primary_name", "fallback_name"),
  [
    (
      lambda: autonomous_module._resolve_autonomous_mcp_gateway_api_key("alice", None),
      "api.agent.autonomous.mcp_config",
      "agent.autonomous.mcp_config",
    ),
  ],
)
def test_gateway_fallbacks_raise_when_primary_dependency_breaks(
  monkeypatch: pytest.MonkeyPatch,
  call: Callable[[], Any],
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
    call()
  assert calls == [primary_name]


def test_addin_dispatch_helper_fallback_raises_when_api_dependency_breaks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  real_import = builtins.__import__

  def fake_import(
    name: str,
    globals_: Any = None,
    locals_: Any = None,
    fromlist: Any = (),
    level: int = 0,
  ) -> Any:
    if name == "api.addin_dispatch":
      raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")
    return real_import(name, globals_, locals_, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", fake_import)

  with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
    excel_dispatch_module._load_addin_dispatch_helpers()


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
