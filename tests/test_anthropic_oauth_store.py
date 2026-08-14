from __future__ import annotations

import io
import os
from pathlib import Path
import stat

import pytest

from agent_gateway import create_agent
from agent_gateway import cli as agent_cli
from agent_gateway.providers.anthropic_oauth import (
  CLAUDE_SETUP_TOKEN_LIFETIME_SECONDS,
  import_claude_setup_token,
  load_anthropic_oauth_record,
  resolve_anthropic_auth_store_path,
  resolve_anthropic_oauth_token,
)
from api.credentials import get_anthropic_config


def test_anthropic_store_is_separate_private_and_one_year(tmp_path: Path) -> None:
  path = tmp_path / "anthropic" / "oauth.json"
  record = import_claude_setup_token("sk-ant-oat01-test", path=path, now=1000)
  assert record["expires_at"] == 1000 + CLAUDE_SETUP_TOKEN_LIFETIME_SECONDS
  assert stat.S_IMODE(path.stat().st_mode) == 0o600
  assert load_anthropic_oauth_record(path) == record


def test_existing_env_token_precedes_gateway_store(tmp_path: Path) -> None:
  path = tmp_path / "oauth.json"
  import_claude_setup_token("sk-ant-oat01-stored", path=path)
  token, resolved_path, record = resolve_anthropic_oauth_token(
    environ={
      "ANTHROPIC_AUTH_STORE_PATH": str(path),
      "ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-existing-session",
      "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-setup-env",
    }
  )
  assert token == "sk-ant-oat01-existing-session"
  assert resolved_path == path
  assert record is None


def test_get_anthropic_config_uses_store_only_when_env_token_absent(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  path = tmp_path / "oauth.json"
  import_claude_setup_token("sk-ant-oat01-stored", path=path)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(path))
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "oauth")
  assert get_anthropic_config()["auth_token"] == "sk-ant-oat01-stored"

  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-current")
  assert get_anthropic_config()["auth_token"] == "sk-ant-oat01-current"


def test_anthropic_cli_import_does_not_modify_claude_login(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  store = tmp_path / "oauth.json"
  subprocess_calls = []
  monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ci-token")
  monkeypatch.setattr(agent_cli.subprocess, "run", lambda *args, **kwargs: subprocess_calls.append(args))

  stdout = io.StringIO()
  code = agent_cli.main(
    ["auth", "login", "anthropic", "--store", str(store)],
    stdout=stdout,
  )
  assert code == 0
  assert subprocess_calls == []
  assert load_anthropic_oauth_record(store)["auth_token"] == "sk-ant-oat01-ci-token"
  assert "Existing Claude Code and gateway sessions were not modified" in stdout.getvalue()

  logout = io.StringIO()
  assert agent_cli.main(
    ["auth", "logout", "anthropic", "--store", str(store)], stdout=logout
  ) == 0
  assert not store.exists()
  assert "Claude Code's saved login was not modified" in logout.getvalue()


def test_anthropic_cli_uses_setup_token_without_logout(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  store = tmp_path / "oauth.json"
  calls = []

  class Result:
    returncode = 0

  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setattr(agent_cli.subprocess, "run", lambda args, check: calls.append(args) or Result())
  monkeypatch.setattr(agent_cli.getpass, "getpass", lambda _prompt: "sk-ant-oat01-pasted")
  assert agent_cli.main(
    ["auth", "login", "anthropic", "--store", str(store)], stdout=io.StringIO()
  ) == 0
  assert calls == [["claude", "setup-token"]]
  assert load_anthropic_oauth_record(store)["auth_token"] == "sk-ant-oat01-pasted"


def test_importing_store_does_not_mutate_existing_gateway_session_config(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
  app = create_agent(
    "test",
    provider="anthropic",
    auth_token="sk-ant-oat01-existing",
  )
  config = app.state.gateway_config
  assert config.service_auth_config_resolver is not None
  [handle] = config.service_provider_handles.values()
  before = dict(config.service_auth_config_resolver(handle).auth_config)
  import_claude_setup_token("sk-ant-oat01-new", path=tmp_path / "oauth.json")
  after = config.service_auth_config_resolver(handle).auth_config
  assert after == before
  assert after["auth_token"] == "sk-ant-oat01-existing"


def test_openai_login_fails_without_changing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
  stderr = io.StringIO()
  assert agent_cli.main(["auth", "login", "openai"], stderr=stderr) == 2
  assert "does not provide ChatGPT subscription OAuth" in stderr.getvalue()
  assert "existing-key" == os.environ["OPENAI_API_KEY"]


def test_codex_login_preserves_existing_login(monkeypatch: pytest.MonkeyPatch) -> None:
  calls = []

  class Result:
    returncode = 0
    stdout = "Logged in using ChatGPT"
    stderr = ""

  monkeypatch.setattr(
    agent_cli.subprocess,
    "run",
    lambda args, **kwargs: calls.append(args) or Result(),
  )
  stdout = io.StringIO()
  assert agent_cli.main(["auth", "login", "codex"], stdout=stdout) == 0
  assert calls == [["codex", "login", "status"]]
  assert "Existing credentials and sessions were not modified" in stdout.getvalue()


def test_codex_login_uses_transactional_cx_enrollment_when_logged_out(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = []

  class Result:
    def __init__(self, returncode):
      self.returncode = returncode
      self.stdout = ""
      self.stderr = ""

  results = iter([Result(1), Result(0)])
  monkeypatch.setattr(
    agent_cli.subprocess,
    "run",
    lambda args, **kwargs: calls.append(args) or next(results),
  )
  assert agent_cli.main(
    [
      "auth", "login", "codex",
      "--profile", "work-account",
      "--email", "work@example.com",
    ],
    stdout=io.StringIO(),
  ) == 0
  assert calls == [
    ["codex", "login", "status"],
    ["cx", "enroll", "work-account", "--email", "work@example.com"],
  ]


def test_codex_logged_out_requires_explicit_safe_enrollment_identity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class Result:
    returncode = 1
    stdout = ""
    stderr = "not logged in"

  monkeypatch.setattr(agent_cli.subprocess, "run", lambda *args, **kwargs: Result())
  stderr = io.StringIO()
  assert agent_cli.main(["auth", "login", "codex"], stderr=stderr) == 2
  assert "requires both --profile and --email" in stderr.getvalue()


def test_store_path_defaults_under_user_data_dir(tmp_path: Path) -> None:
  assert resolve_anthropic_auth_store_path(environ={"USER_DATA_DIR": str(tmp_path)}) == (
    tmp_path / "anthropic" / "oauth.json"
  )
