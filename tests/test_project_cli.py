from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_gateway import cli as agent_cli
from agent_gateway import project
from agent_gateway.project import AgentProjectError


def test_agent_init_writes_working_project(tmp_path: Path) -> None:
  stdout = io.StringIO()

  code = agent_cli.main(["init", "demo-agent", "--dir", str(tmp_path / "demo")], stdout=stdout)

  assert code == 0
  root = tmp_path / "demo"
  payload = yaml.safe_load((root / "agent.yaml").read_text(encoding="utf-8"))
  assert payload["name"] == "demo-agent"
  assert payload["provider"] == "anthropic"
  assert payload["skills_dir"] == "skills"
  assert payload["skill_state_file"] == "skill_state.json"
  assert (root / "agent.py").read_text(encoding="utf-8") == (
    "from agent_gateway.project import create_agent_from_yaml\n\n"
    "app = create_agent_from_yaml(\"agent.yaml\")\n"
  )
  skill = (root / "skills" / "general.md").read_text(encoding="utf-8")
  assert "agent_callable: true" in skill
  assert "persist_state: true" in skill
  assert (root / "skills" / "general.md").exists()
  assert "Created agent project" in stdout.getvalue()


def test_agent_init_refuses_to_overwrite_generated_files(tmp_path: Path) -> None:
  root = tmp_path / "demo"
  agent_cli.main(["init", "demo", "--dir", str(root)], stdout=io.StringIO())

  stderr = io.StringIO()
  code = agent_cli.main(["init", "demo", "--dir", str(root)], stdout=io.StringIO(), stderr=stderr)

  assert code == 2
  assert "Refusing to overwrite existing files" in stderr.getvalue()


def test_create_agent_from_yaml_forwards_resolved_config(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  config_path = tmp_path / "agent.yaml"
  config_path.write_text(
    yaml.safe_dump(
      {
        "name": "demo",
        "system_prompt": "Use tools carefully.",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "host": "localhost",
        "port": 9000,
        "api_prefix": "v2",
        "skills_dir": "skills",
        "skill_state_file": "skill_state.json",
        "outputs_dir": "artifacts",
        "mcp_servers": {"fs": {"command": "npx", "args": ["-y", "server", "."]}},
        "code_execution": True,
        "max_tokens": 2048,
        "max_turns": 8,
        "max_budget_usd": 1.5,
        "per_turn_timeout": 120,
      },
      sort_keys=False,
    ),
    encoding="utf-8",
  )
  calls: list[dict[str, Any]] = []
  sentinel = object()

  def fake_create_agent(system_prompt: str, **kwargs: Any) -> object:
    calls.append({"system_prompt": system_prompt, **kwargs})
    return sentinel

  monkeypatch.setattr(project, "create_agent", fake_create_agent)

  app = project.create_agent_from_yaml(config_path)

  assert app is sentinel
  assert calls == [
    {
      "system_prompt": "Use tools carefully.",
      "provider": "openai",
      "model": "gpt-4o-mini",
      "max_tokens": 2048,
      "mcp_servers": {"fs": {"command": "npx", "args": ["-y", "server", "."]}},
      "skills_dir": tmp_path / "skills",
      "skill_state_file": tmp_path / "skill_state.json",
      "outputs_dir": tmp_path / "artifacts",
      "code_execution": True,
      "max_turns": 8,
      "max_budget_usd": 1.5,
      "per_turn_timeout": 120,
      "prefix": "/v2",
    }
  ]


def test_agent_add_mcp_and_provider_update_config(tmp_path: Path) -> None:
  config_path = tmp_path / "agent.yaml"
  project.save_agent_project_payload(
    project.default_agent_config_payload(name="demo"),
    config_path,
  )

  mcp_stdout = io.StringIO()
  provider_stdout = io.StringIO()
  mcp_code = agent_cli.main(
    [
      "add",
      "mcp",
      "filesystem",
      "npx",
      "-y",
      "@modelcontextprotocol/server-filesystem",
      ".",
      "--config",
      str(config_path),
    ],
    stdout=mcp_stdout,
  )
  provider_code = agent_cli.main(
    ["add", "provider", "openai", "--model", "gpt-4o-mini", "--config", str(config_path)],
    stdout=provider_stdout,
  )

  assert mcp_code == 0
  assert provider_code == 0
  payload = project.load_agent_project_payload(config_path)
  assert payload["provider"] == "openai"
  assert payload["model"] == "gpt-4o-mini"
  assert payload["mcp_servers"]["filesystem"] == {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
  }


def test_agent_run_invokes_uvicorn_with_env_config(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  config_path = tmp_path / "agent.yaml"
  project.save_agent_project_payload(
    {
      **project.default_agent_config_payload(name="demo"),
      "host": "0.0.0.0",
      "port": 8123,
      "api_prefix": "/gateway",
    },
    config_path,
  )
  monkeypatch.setenv(project.PROJECT_CONFIG_ENV, "previous.yaml")
  calls: list[dict[str, Any]] = []

  def fake_run(app: str, **kwargs: Any) -> None:
    calls.append({"app": app, **kwargs, "env": os.environ[project.PROJECT_CONFIG_ENV]})

  stdout = io.StringIO()
  code = agent_cli.main(
    ["run", "--config", str(config_path), "--reload"],
    stdout=stdout,
    uvicorn_run=fake_run,
  )

  assert code == 0
  assert calls == [
    {
      "app": "agent_gateway.project:create_app_from_env",
      "factory": True,
      "host": "0.0.0.0",
      "port": 8123,
      "reload": True,
      "reload_dirs": [str(tmp_path)],
      "env": str(config_path),
    }
  ]
  assert os.environ[project.PROJECT_CONFIG_ENV] == "previous.yaml"
  assert "http://0.0.0.0:8123/gateway" in stdout.getvalue()


def test_project_config_rejects_unknown_keys(tmp_path: Path) -> None:
  config_path = tmp_path / "agent.yaml"
  config_path.write_text("name: demo\nsurprise: true\n", encoding="utf-8")

  with pytest.raises(AgentProjectError, match="Unsupported agent config keys: surprise"):
    project.load_agent_project_config(config_path)


def test_launch_project_examples_have_loadable_configs() -> None:
  examples_root = Path(__file__).resolve().parents[1] / "examples"
  for example_name in ("10-daily-briefing", "11-research-report"):
    root = examples_root / example_name
    config = project.load_agent_project_config(root / "agent.yaml")

    assert config.name
    assert config.skills_dir == root / "skills"
    assert config.skill_state_file == root / "skill_state.json"
    assert "filesystem" in config.mcp_servers
    assert (root / "README.md").is_file()
    assert (root / "agent.py").is_file()
    assert list((root / "skills").glob("*.md"))
