from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

def _offline_wheel_python(tmp_path: Path) -> Path:
  """A pip that builds these wheels without an index.

  Build isolation fetches setuptools. The EC2 gate host has no index: `pip wheel`
  on the base interpreter exited 1 after about six minutes, and the JUnit did not
  keep the captured stderr. A fresh venv has no checkout pip.conf. The caller's
  setuptools and wheel are copied in, as the model-authority smoke test does,
  and the build uses `--no-build-isolation`.
  """
  venv = tmp_path / "build-venv"
  created = subprocess.run(
    [sys.executable, "-m", "venv", str(venv)],
    check=False, capture_output=True, text=True,
  )
  assert created.returncode == 0, created.stderr
  python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
  site = Path(subprocess.run(
    [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
    check=True, capture_output=True, text=True,
  ).stdout.strip())
  for name in ("setuptools", "wheel"):
    distribution = importlib.metadata.distribution(name)
    for file in distribution.files or ():
      if not file.parts or file.parts[0] == "..":
        continue
      source = Path(str(distribution.locate_file(file)))
      if not source.is_file():
        continue
      target = site / file
      target.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source, target)
  return python


def _wheel_env(tmp_path: Path) -> dict[str, str]:
  environment = {
    name: value for name, value in os.environ.items()
    if name != "PYTHONPATH" and not name.startswith("PIP_")
  }
  environment["PIP_NO_INDEX"] = "1"
  environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
  environment["PIP_CACHE_DIR"] = str(tmp_path / "pip-cache")
  return environment


def test_installed_wheel_owns_sessionless_role_policy_without_host_api(
  tmp_path: Path,
) -> None:
  source_root = Path(__file__).resolve().parents[1]
  package_root = tmp_path / "agent-gateway"
  package_root.mkdir()
  for filename in ("pyproject.toml", "README.md", "LICENSE"):
    shutil.copy2(source_root / filename, package_root / filename)
  shutil.copytree(
    source_root / "agent_gateway",
    package_root / "agent_gateway",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
  )
  contracts_source = source_root.parent / "agent-workflow-contracts"
  contracts_root = tmp_path / "agent-workflow-contracts"
  contracts_root.mkdir()
  shutil.copy2(contracts_source / "pyproject.toml", contracts_root / "pyproject.toml")
  shutil.copytree(
    contracts_source / "agent_workflow_contracts",
    contracts_root / "agent_workflow_contracts",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
  )
  wheel_dir = tmp_path / "wheel"
  wheel_dir.mkdir()
  builder_python = _offline_wheel_python(tmp_path)
  build_env = _wheel_env(tmp_path)
  built = subprocess.run(
    [
      str(builder_python),
      "-I",
      "-m",
      "pip",
      "wheel",
      "--no-deps",
      "--no-build-isolation",
      "--wheel-dir",
      str(wheel_dir),
      str(package_root),
      str(contracts_root),
    ],
    check=False,
    env=build_env,
    capture_output=True,
    text=True,
  )
  assert built.returncode == 0, built.stderr
  wheel = next(wheel_dir.glob("ai_agent_gateway-*.whl"))
  contracts_wheel = next(wheel_dir.glob("agent_workflow_contracts-*.whl"))
  with ZipFile(wheel) as archive:
    assert not any(
      name.startswith("agent_workflow_contracts/") for name in archive.namelist()
    )
  installed = tmp_path / "installed"
  installed_wheel = subprocess.run(
    [
      str(builder_python),
      "-I",
      "-m",
      "pip",
      "install",
      "--no-deps",
      "--no-index",
      "--disable-pip-version-check",
      "--target",
      str(installed),
      str(wheel),
      str(contracts_wheel),
    ],
    check=False,
    env=build_env,
    capture_output=True,
    text=True,
  )
  assert installed_wheel.returncode == 0, installed_wheel.stderr

  script = """
import asyncio
import json
from pathlib import Path
import sys

installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))

import agent_gateway
import agent_workflow_contracts
from agent_gateway import ToolDispatcher
from agent_gateway.approval_policy import RunContext
from agent_gateway.registry_scope import RegistryScope
from hank_model_transport import policy_imports
from model_authority.bind import CapabilityBind

TEST_CAPABILITY_BIND = CapabilityBind(
  schema_version="1.0",
  capability_id="session.driver",
  model_key="anthropic.test-autonomous",
  provider="anthropic",
  upstream_model="claude-test",
  adapter="anthropic.messages",
  protocol_profile="messages.standard",
  route="anthropic.public",
  effort="high",
  credential_principal="service",
  credential_ref="autonomous-service:test-handle",
  run_mode="autonomous",
  registry_revision="test-registry-1",
  policy_revision="test-policy-1",
  selection_source="capability_default",
)

RUN_CONTEXT = RunContext(
  user_id="test-user",
  request_id="test-request",
  registry_scope=RegistryScope("run", "test-run", "/tmp/test-store"),
)


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
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=Mcp(),
      local_tool_handlers={"rw_list_studies": handler},
      role=role,
      local_tool_class_resolver=local_tool_class,
      local_catalog_action_resolver=local_catalog_action,
      run_context=RUN_CONTEXT,
    )
    result, error = await dispatcher.dispatch(
      "call-" + str(role),
      "rw_list_studies",
      {},
      advertised_tool_names={"rw_list_studies"},
    )
    outcomes[str(role)] = {"result": result, "error": error}

  owner_dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=Mcp(),
    local_tool_handlers={"rw_list_studies": handler},
    role="owner",
    local_tool_class_resolver=local_tool_class,
    local_catalog_action_resolver=local_catalog_action,
    run_context=RUN_CONTEXT,
  )
  excluded_result, excluded_error = await owner_dispatcher.dispatch(
    "call-excluded",
    "rw_delete_study",
    {},
    advertised_tool_names={"rw_delete_study"},
  )

  print(json.dumps({
    "agent_gateway_file": str(Path(agent_gateway.__file__).resolve()),
    "agent_workflow_contracts_file": str(Path(agent_workflow_contracts.__file__).resolve()),
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
  assert Path(observed["agent_workflow_contracts_file"]).is_relative_to(installed)
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
