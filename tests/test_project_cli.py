from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import httpx
import yaml

from agent_gateway import cli as agent_cli
from agent_gateway import project
from agent_gateway.project import AgentProjectError
from agent_gateway.providers import xai_oauth


def test_agent_init_writes_working_project(tmp_path: Path) -> None:
  stdout = io.StringIO()

  code = agent_cli.main(["init", "demo-agent", "--dir", str(tmp_path / "demo")], stdout=stdout)

  assert code == 0
  root = tmp_path / "demo"
  payload = yaml.safe_load((root / "agent.yaml").read_text(encoding="utf-8"))
  assert payload["name"] == "demo-agent"
  assert "provider" not in payload
  assert "model" not in payload
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
        "model_key": "openai.gpt-5-6",
        "effort": "high",
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
      "model_key": "openai.gpt-5-6",
      "effort": "high",
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


def test_agent_add_mcp_and_stable_model_update_config(tmp_path: Path) -> None:
  config_path = tmp_path / "agent.yaml"
  project.save_agent_project_payload(
    project.default_agent_config_payload(name="demo"),
    config_path,
  )

  mcp_stdout = io.StringIO()
  model_stdout = io.StringIO()
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
  model_code = agent_cli.main(
    [
      "add",
      "model",
      "openai.gpt-5-6",
      "--effort",
      "high",
      "--config",
      str(config_path),
    ],
    stdout=model_stdout,
  )

  assert mcp_code == 0
  assert model_code == 0
  payload = project.load_agent_project_payload(config_path)
  assert payload["model_key"] == "openai.gpt-5-6"
  assert payload["effort"] == "high"
  assert "provider" not in payload
  assert "model" not in payload
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


def test_agent_xai_auth_login_status_and_logout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"

  async def fake_login(*, config, on_verification, client=None):
    device = xai_oauth.XAIDeviceCode(
      "device-secret",
      "ABCD-1234",
      "https://accounts.x.ai/oauth2/device",
      None,
      900,
      5,
    )
    await on_verification(device)
    record = {
      "access_token": "access-1",
      "refresh_token": "refresh-1",
      "expires_at": 4_000_000_000,
      "scope": xai_oauth.DEFAULT_XAI_OAUTH_SCOPE,
      "issuer": xai_oauth.DEFAULT_XAI_OAUTH_ISSUER,
      "client_id": xai_oauth.DEFAULT_XAI_OAUTH_CLIENT_ID,
    }
    xai_oauth.save_xai_token_record(Path(config["auth_store_path"]), record)
    return record, Path(config["auth_store_path"])

  monkeypatch.setattr(xai_oauth, "login_xai_device_code", fake_login)
  stdout = io.StringIO()
  assert agent_cli.main(
    ["auth", "login", "xai", "--no-browser", "--store", str(store)], stdout=stdout
  ) == 0
  output = stdout.getvalue()
  assert "ABCD-1234" in output
  assert "device-secret" not in output
  assert "login complete" in output

  status = io.StringIO()
  assert agent_cli.main(["auth", "status", "xai", "--store", str(store)], stdout=status) == 0
  # `auth status` is a local-only read that never contacts xAI, so it reports what it verified
  # (a stored, unexpired record) — not "active", which would over-claim a credential the server
  # may have revoked. See cli.py _auth_status_xai.
  assert "stored; not server-validated" in status.getvalue()
  assert "permissions=0600" in status.getvalue()

  logout = io.StringIO()
  assert agent_cli.main(["auth", "logout", "xai", "--store", str(store)], stdout=logout) == 0
  assert store.exists()
  assert xai_oauth.load_xai_token_record(store) == {"reauth_required": True}
  logged_out_status = io.StringIO()
  assert agent_cli.main(
    ["auth", "status", "xai", "--store", str(store)],
    stdout=logged_out_status,
  ) == 1
  assert "re-authentication required" in logged_out_status.getvalue()


def test_xai_auth_status_refresh_pending_is_nonzero(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  xai_oauth.save_xai_token_record(store, {
    "access_token": "access-1",
    "refresh_token": "refresh-1",
    "expires_at": 4_000_000_000,
    "scope": xai_oauth.DEFAULT_XAI_OAUTH_SCOPE,
    "issuer": xai_oauth.DEFAULT_XAI_OAUTH_ISSUER,
    "client_id": xai_oauth.DEFAULT_XAI_OAUTH_CLIENT_ID,
    "refresh_pending": True,
  })

  status = io.StringIO()
  assert agent_cli.main(
    ["auth", "status", "xai", "--store", str(store)],
    stdout=status,
  ) == 1
  assert "previous refresh did not complete" in status.getvalue()
  assert "re-authentication required" in status.getvalue()


def test_xai_logout_uses_store_lock_and_writes_tombstone(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  store = tmp_path / "oauth.json"
  cached_record = {
    "access_token": "access-1",
    "refresh_token": "refresh-1",
    "expires_at": 4_000_000_000,
    "scope": xai_oauth.DEFAULT_XAI_OAUTH_SCOPE,
    "issuer": xai_oauth.DEFAULT_XAI_OAUTH_ISSUER,
    "client_id": xai_oauth.DEFAULT_XAI_OAUTH_CLIENT_ID,
    "token_endpoint": "https://auth.x.ai/oauth2/token",
  }
  xai_oauth.save_xai_token_record(store, cached_record)
  real_lock = xai_oauth._store_lock
  lock_entries: list[Path] = []

  @asynccontextmanager
  async def observed_lock(path: Path):
    lock_entries.append(path)
    async with real_lock(path):
      yield

  monkeypatch.setattr(xai_oauth, "_store_lock", observed_lock)
  assert agent_cli.main(
    ["auth", "logout", "xai", "--store", str(store)],
    stdout=io.StringIO(),
  ) == 0
  assert lock_entries == [store]
  assert xai_oauth.load_xai_token_record(store) == {"reauth_required": True}
  assert store.stat().st_mode & 0o777 == 0o600
  posts: list[str] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    posts.append("unexpected")
    return httpx.Response(500)

  async def refresh_cached() -> None:
    settings = xai_oauth.resolve_xai_oauth_settings({"auth_store_path": str(store)})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await xai_oauth.refresh_xai_oauth_token(
        cached_record,
        settings=settings,
        client=client,
        force=True,
      )

  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    asyncio.run(refresh_cached())
  assert posts == []


def test_anthropic_logout_still_unlinks_store(tmp_path: Path) -> None:
  store = tmp_path / "anthropic-oauth.json"
  store.write_text('{"access_token": "token"}\n', encoding="utf-8")

  assert agent_cli.main(
    ["auth", "logout", "anthropic", "--store", str(store)],
    stdout=io.StringIO(),
  ) == 0
  assert not store.exists()


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
