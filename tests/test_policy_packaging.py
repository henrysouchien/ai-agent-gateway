from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_installed_wheel_owns_sessionless_role_policy_without_host_api(
  tmp_path: Path,
) -> None:
  package_root = Path(__file__).resolve().parents[1]
  wheel_dir = tmp_path / "wheel"
  wheel_dir.mkdir()
  builder_python = Path(sys.base_prefix) / (
    "python.exe" if os.name == "nt" else "bin/python3"
  )
  subprocess.run(
    [
      str(builder_python),
      "-m",
      "pip",
      "wheel",
      "--no-deps",
      "--wheel-dir",
      str(wheel_dir),
      str(package_root),
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  wheel = next(wheel_dir.glob("ai_agent_gateway-*.whl"))
  installed = tmp_path / "installed"
  subprocess.run(
    [
      str(builder_python),
      "-m",
      "pip",
      "install",
      "--no-deps",
      "--disable-pip-version-check",
      "--target",
      str(installed),
      str(wheel),
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  script = """
import asyncio
import json
from pathlib import Path
import sys

installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))

import agent_gateway
from agent_gateway import ToolDispatcher
from agent_gateway import policy_imports


class Mcp:
  def is_mcp_tool(self, _name):
    return False

  def get_server_for_tool(self, _name):
    return None


async def main():
  calls = []
  exact_tools = frozenset({"rw_list_studies"})

  def local_tool_class(tool_name):
    if tool_name not in exact_tools:
      raise ValueError("tool is outside the exact local surface")
    return "state_write"

  def local_catalog_action(tool_name):
    if tool_name not in exact_tools:
      raise ValueError("tool is outside the exact local surface")
    return None

  async def handler(tool_input, **_kwargs):
    calls.append(dict(tool_input))
    return {"executed": True}, None

  outcomes = {}
  for role in ("owner", "invite", None):
    dispatcher = ToolDispatcher(
      mcp_client=Mcp(),
      local_tool_handlers={"rw_list_studies": handler},
      role=role,
      local_tool_class_resolver=local_tool_class,
      local_catalog_action_resolver=local_catalog_action,
    )
    result, error = await dispatcher.dispatch(
      "call-" + str(role),
      "rw_list_studies",
      {},
      advertised_tool_names={"rw_list_studies"},
    )
    outcomes[str(role)] = {"result": result, "error": error}

  owner_dispatcher = ToolDispatcher(
    mcp_client=Mcp(),
    local_tool_handlers={"rw_list_studies": handler},
    role="owner",
    local_tool_class_resolver=local_tool_class,
    local_catalog_action_resolver=local_catalog_action,
  )
  excluded_result, excluded_error = await owner_dispatcher.dispatch(
    "call-excluded",
    "rw_delete_study",
    {},
    advertised_tool_names={"rw_delete_study"},
  )

  print(json.dumps({
    "agent_gateway_file": str(Path(agent_gateway.__file__).resolve()),
    "calls": calls,
    "excluded": {"result": excluded_result, "error": excluded_error},
    "host_policy_available": policy_imports.load_server_policy_module() is not None,
    "host_catalog_imported": any(
      name in sys.modules for name in ("fms.action_catalog", "api.fms.action_catalog")
    ),
    "outcomes": outcomes,
  }, sort_keys=True))


asyncio.run(main())
"""
  isolated_env = os.environ.copy()
  isolated_env.pop("PYTHONPATH", None)
  completed = subprocess.run(
    [sys.executable, "-I", "-c", script, str(installed)],
    cwd=tmp_path,
    env=isolated_env,
    check=True,
    capture_output=True,
    text=True,
  )
  observed = json.loads(completed.stdout)

  assert Path(observed["agent_gateway_file"]).is_relative_to(installed)
  assert observed["host_policy_available"] is False
  assert observed["host_catalog_imported"] is False
  assert observed["calls"] == [{}]
  assert observed["excluded"]["result"] is None
  assert observed["excluded"]["error"]["code"] == "unknown_tool"
  assert observed["outcomes"]["owner"] == {
    "error": None,
    "result": {"executed": True},
  }
  for role in ("invite", "None"):
    assert observed["outcomes"][role]["result"] is None
    assert observed["outcomes"][role]["error"]["code"] == "role_policy_denied"
