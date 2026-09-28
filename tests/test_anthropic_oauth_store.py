from __future__ import annotations

import io
import os
from pathlib import Path
import stat

import pytest

from agent_gateway import create_agent
from agent_gateway import cli as agent_cli
from agent_gateway.providers import anthropic_oauth
from agent_gateway.providers.anthropic_oauth import (
  ANTHROPIC_OAUTH_CLIENT_ID,
  ANTHROPIC_USER_SCOPE_FIELD,
  AnthropicOAuthError,
  AnthropicOAuthRecord,
  complete_anthropic_login,
  enroll_anthropic_setup_token,
  load_anthropic_oauth_store,
  parse_anthropic_callback,
  refresh_anthropic_oauth_record,
  resolve_anthropic_auth_store_path,
  resolve_anthropic_credentials,
  select_anthropic_credential,
  start_anthropic_login,
  upsert_anthropic_oauth_record,
)


def _record(identity: str, *, expires_at: float = 9_999_999_999.0) -> AnthropicOAuthRecord:
  return AnthropicOAuthRecord(
    identity=identity,
    access_token=f"sk-ant-oat01-{identity}-access",
    refresh_token=f"sk-ant-ort01-{identity}-refresh",
    expires_at=expires_at,
    created_at=1000.0,
    updated_at=1000.0,
  )


def test_store_round_trips_two_accounts_privately(tmp_path: Path) -> None:
  path = tmp_path / "anthropic" / "oauth.json"
  upsert_anthropic_oauth_record(path, _record("first@example.com"))
  upsert_anthropic_oauth_record(path, _record("second@example.com"))

  assert stat.S_IMODE(path.stat().st_mode) == 0o600
  records = load_anthropic_oauth_store(path)
  assert [record.identity for record in records] == ["first@example.com", "second@example.com"]
  assert records[1].refresh_token == "sk-ant-ort01-second@example.com-refresh"

  # Re-enrolling one account replaces that record in place and leaves the
  # sibling, and the enrollment order, untouched.
  upsert_anthropic_oauth_record(
    path,
    AnthropicOAuthRecord(
      identity="first@example.com",
      access_token="sk-ant-oat01-first-rotated",
      refresh_token="sk-ant-ort01-first-rotated",
      expires_at=123.0,
    ),
  )
  records = load_anthropic_oauth_store(path)
  assert [record.identity for record in records] == ["first@example.com", "second@example.com"]
  assert records[0].access_token == "sk-ant-oat01-first-rotated"
  assert records[1].access_token == "sk-ant-oat01-second@example.com-access"


def test_env_token_precedes_enrolled_accounts_and_both_are_siblings(tmp_path: Path) -> None:
  path = tmp_path / "oauth.json"
  upsert_anthropic_oauth_record(path, _record("stored@example.com"))
  sources = resolve_anthropic_credentials(
    environ={
      "ANTHROPIC_AUTH_STORE_PATH": str(path),
      "ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-existing-session",
      "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-setup-env",
    }
  )
  assert sources.store_path == path
  assert sources.tokens == (
    "sk-ant-oat01-existing-session",
    "sk-ant-oat01-setup-env",
    "sk-ant-oat01-stored@example.com-access",
  )
  # The enrolled account is keyed by identity so a refreshed access token keeps
  # its parking window; environment tokens are their own keys.
  assert sources.keys[-1] == "stored@example.com"
  env_credential = sources.by_token("sk-ant-oat01-existing-session")
  assert env_credential is not None
  assert env_credential.record is None


def test_env_token_that_is_an_enrolled_access_token_resolves_to_that_account(
  tmp_path: Path,
) -> None:
  path = tmp_path / "oauth.json"
  upsert_anthropic_oauth_record(path, _record("shared@example.com"))
  sources = resolve_anthropic_credentials(
    environ={
      "ANTHROPIC_AUTH_STORE_PATH": str(path),
      "ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-shared@example.com-access",
    }
  )
  assert len(sources.credentials) == 1
  assert sources.credentials[0].key == "shared@example.com"
  assert sources.credentials[0].record is not None


def test_legacy_single_record_store_still_serves_as_one_account(tmp_path: Path) -> None:
  path = tmp_path / "oauth.json"
  path.write_text(
    '{"auth_token": "sk-ant-oat01-legacy", "expires_at": 4000.0, '
    '"source": "claude-setup-token"}\n',
    encoding="utf-8",
  )
  sources = resolve_anthropic_credentials(
    environ={"ANTHROPIC_AUTH_STORE_PATH": str(path)}
  )
  assert sources.tokens == ("sk-ant-oat01-legacy",)
  record = sources.credentials[0].record
  assert record is not None
  # No refresh grant: it is used as-is and never refreshed.
  assert record.refresh_token == ""
  assert record.needs_refresh(now=9_999_999_999.0) is False


def test_login_enrolls_the_account_the_token_response_names(tmp_path: Path) -> None:
  path = tmp_path / "oauth.json"
  request = start_anthropic_login()
  posted: list[tuple[str, dict]] = []

  def post(url, body):
    posted.append((url, dict(body)))
    return {
      "access_token": "sk-ant-oat01-new",
      "refresh_token": "sk-ant-ort01-new",
      "expires_in": 28800,
      "scope": "user:profile user:inference",
      "account": {"email_address": "henry@example.com", "uuid": "acct-1"},
    }

  record = complete_anthropic_login(
    request,
    f"auth-code-1#{request.state}",
    store_path=path,
    now=1000.0,
    post=post,
  )
  assert record.identity == "henry@example.com"
  assert record.expires_at == 1000.0 + 28800
  _url, body = posted[0]
  assert body["grant_type"] == "authorization_code"
  assert body["code"] == "auth-code-1"
  assert body["code_verifier"] == request.code_verifier
  assert [stored.identity for stored in load_anthropic_oauth_store(path)] == [
    "henry@example.com"
  ]


def test_login_refuses_a_code_bound_to_another_request(tmp_path: Path) -> None:
  path = tmp_path / "oauth.json"
  request = start_anthropic_login()

  def post(url, body):  # pragma: no cover - must not be reached
    raise AssertionError("the code must not be exchanged when the state mismatches")

  with pytest.raises(AnthropicOAuthError, match="state does not match"):
    complete_anthropic_login(
      request, "auth-code-1#someone-elses-state", store_path=path, post=post
    )
  assert load_anthropic_oauth_store(path) == ()


def test_login_without_a_refresh_token_is_not_enrolled(tmp_path: Path) -> None:
  path = tmp_path / "oauth.json"
  request = start_anthropic_login()

  with pytest.raises(AnthropicOAuthError, match="no refresh token"):
    complete_anthropic_login(
      request,
      f"code#{request.state}",
      store_path=path,
      post=lambda url, body: {"access_token": "sk-ant-oat01-x", "expires_in": 3600},
    )
  assert load_anthropic_oauth_store(path) == ()


def test_login_identity_falls_back_to_the_profile_endpoint(tmp_path: Path) -> None:
  path = tmp_path / "oauth.json"
  request = start_anthropic_login()
  record = complete_anthropic_login(
    request,
    f"code#{request.state}",
    store_path=path,
    now=500.0,
    post=lambda url, body: {
      "access_token": "sk-ant-oat01-p",
      "refresh_token": "sk-ant-ort01-p",
      "expires_in": 100,
    },
    get_profile=lambda url, token: {"account": {"uuid": "acct-42"}},
  )
  assert record.identity == "acct-42"


def test_callback_paste_forms() -> None:
  assert parse_anthropic_callback("code-1#state-1") == ("code-1", "state-1")
  assert parse_anthropic_callback(
    "https://console.anthropic.com/oauth/code/callback?code=code-2&state=state-2"
  ) == ("code-2", "state-2")
  assert parse_anthropic_callback("?code=code-3&state=state-3") == ("code-3", "state-3")
  assert parse_anthropic_callback("  bare-code  ") == ("bare-code", "")


def test_anthropic_cli_login_enrolls_lists_and_removes_one_account(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  store = tmp_path / "oauth.json"
  opened: list[str] = []
  monkeypatch.setattr(agent_cli.webbrowser, "open", lambda url: opened.append(url))
  monkeypatch.setattr(
    agent_cli.getpass,
    "getpass",
    lambda _prompt: "code-9#" + opened[0].split("state=")[1],
  )
  monkeypatch.setattr(
    anthropic_oauth,
    "_post_token_endpoint",
    lambda url, body: {
      "access_token": "sk-ant-oat01-cli-access-token-value",
      "refresh_token": "sk-ant-ort01-cli",
      "expires_in": 28800,
      "account": {"email_address": "cli@example.com"},
    },
  )

  stdout = io.StringIO()
  assert agent_cli.main(
    ["auth", "login", "anthropic", "--store", str(store)], stdout=stdout
  ) == 0
  assert opened and opened[0].startswith("https://claude.ai/oauth/authorize?")
  assert "Enrolled cli@example.com" in stdout.getvalue()
  assert "Existing Claude Code and gateway sessions were not modified" in stdout.getvalue()

  listed = io.StringIO()
  assert agent_cli.main(
    ["auth", "list", "anthropic", "--store", str(store)], stdout=listed
  ) == 0
  # Masked: the operator sees which account and which credential, not the secret.
  assert "cli@example.com" in listed.getvalue()
  assert "sk-ant-oat01-cli-access-token-value" not in listed.getvalue()
  assert "sk-ant-oat01…alue" in listed.getvalue()

  removed = io.StringIO()
  assert agent_cli.main(
    ["auth", "remove", "anthropic", "cli@example.com", "--store", str(store)],
    stdout=removed,
  ) == 0
  assert load_anthropic_oauth_store(store) == ()
  assert agent_cli.main(
    ["auth", "list", "anthropic", "--store", str(store)], stdout=io.StringIO()
  ) == 1


def test_enrolling_an_account_does_not_mutate_an_existing_gateway_session_config(
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
  upsert_anthropic_oauth_record(tmp_path / "oauth.json", _record("new@example.com"))
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


def test_expired_account_refreshes_before_use_and_persists_the_rotated_token(
  tmp_path: Path,
) -> None:
  path = tmp_path / "oauth.json"
  upsert_anthropic_oauth_record(path, _record("stale@example.com", expires_at=500.0))
  bodies: list[dict] = []

  def post(url, body):
    bodies.append(dict(body))
    return {
      "access_token": "sk-ant-oat01-rotated-access",
      "refresh_token": "sk-ant-ort01-rotated-refresh",
      "expires_in": 28800,
    }

  sources = resolve_anthropic_credentials(
    environ={"ANTHROPIC_AUTH_STORE_PATH": str(path)}
  )
  record = sources.credentials[0].record
  assert record is not None and record.needs_refresh(now=1000.0)
  refreshed = refresh_anthropic_oauth_record(
    record, store_path=path, now=1000.0, post=post
  )

  assert refreshed.access_token == "sk-ant-oat01-rotated-access"
  assert refreshed.expires_at == 1000.0 + 28800
  assert bodies == [
    {
      "grant_type": "refresh_token",
      "refresh_token": "sk-ant-ort01-stale@example.com-refresh",
      "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
    }
  ]
  # The rotated refresh token is the one on disk: presenting the previous one
  # again would be rejected and would strand the account.
  [stored] = load_anthropic_oauth_store(path)
  assert stored.refresh_token == "sk-ant-ort01-rotated-refresh"
  assert stored.access_token == "sk-ant-oat01-rotated-access"
  assert stored.last_error == ""


def test_selection_refreshes_the_expired_account_it_is_about_to_bind(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  path = tmp_path / "oauth.json"
  upsert_anthropic_oauth_record(path, _record("stale@example.com", expires_at=0.0))
  monkeypatch.setattr(
    anthropic_oauth,
    "_post_token_endpoint",
    lambda url, body: {
      "access_token": "sk-ant-oat01-bound-fresh",
      "refresh_token": "sk-ant-ort01-bound-fresh",
      "expires_in": 28800,
    },
  )
  sources = resolve_anthropic_credentials(
    environ={"ANTHROPIC_AUTH_STORE_PATH": str(path)}
  )

  pool = anthropic_oauth.AnthropicCredentialPool()
  assert select_anthropic_credential(sources, pool=pool) == "sk-ant-oat01-bound-fresh"


def test_failed_refresh_marks_the_account_and_serves_the_next_sibling(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  path = tmp_path / "oauth.json"
  upsert_anthropic_oauth_record(path, _record("stale@example.com", expires_at=0.0))
  upsert_anthropic_oauth_record(path, _record("healthy@example.com"))

  def failing_post(url, body):
    raise AnthropicOAuthError("Anthropic OAuth token request failed (400): invalid_grant")

  monkeypatch.setattr(anthropic_oauth, "_post_token_endpoint", failing_post)
  sources = resolve_anthropic_credentials(
    environ={"ANTHROPIC_AUTH_STORE_PATH": str(path)}
  )
  pool = anthropic_oauth.AnthropicCredentialPool()

  assert select_anthropic_credential(sources, pool=pool) == (
    "sk-ant-oat01-healthy@example.com-access"
  )
  stale, healthy = load_anthropic_oauth_store(path)
  # The record is marked, never deleted: re-enrolling is the operator's remedy.
  assert stale.identity == "stale@example.com"
  assert "invalid_grant" in stale.last_error
  assert stale.refresh_token == "sk-ant-ort01-stale@example.com-refresh"
  assert healthy.last_error == ""
  assert pool.blocked_until("stale@example.com") > 0.0


def test_user_store_lives_under_that_users_private_data_root(tmp_path: Path) -> None:
  environ = {"USER_DATA_DIR": str(tmp_path)}

  assert resolve_anthropic_auth_store_path(environ=environ, user_scope=1) == (
    tmp_path / "users" / "1" / "anthropic" / "oauth.json"
  )
  # The scope is a property of the credential material, so a bound credential
  # finds the same store without being told the identity a second time.
  assert resolve_anthropic_auth_store_path(
    {ANTHROPIC_USER_SCOPE_FIELD: "1"}, environ=environ
  ) == (tmp_path / "users" / "1" / "anthropic" / "oauth.json")
  # No user: the org store every user's pool ends with.
  assert resolve_anthropic_auth_store_path(environ=environ) == (
    tmp_path / "anthropic" / "oauth.json"
  )


def test_a_users_pool_is_their_accounts_then_the_org_and_never_another_users(
  tmp_path: Path,
) -> None:
  """The pool order, stated as one derivation and checked for both users.

  Henry's personal Claude subscription may serve his own turns and nothing
  else: a credential enrolled under `risk_user_id` 1 must never appear in the
  pool resolved for user 2, and user 2 — who has enrolled nothing — must see
  exactly the org credentials the deployment configured.
  """

  environ = {
    "USER_DATA_DIR": str(tmp_path),
    "ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-org-primary",
  }
  upsert_anthropic_oauth_record(
    resolve_anthropic_auth_store_path(environ=environ, user_scope=1),
    _record("henry@example.com"),
  )
  upsert_anthropic_oauth_record(
    resolve_anthropic_auth_store_path(environ=environ),
    _record("org@example.com"),
  )

  first = resolve_anthropic_credentials(environ=environ, user_scope=1)
  assert first.keys == ("henry@example.com", "sk-ant-oat01-org-primary", "org@example.com")

  second = resolve_anthropic_credentials(environ=environ, user_scope=2)
  assert second.keys == ("sk-ant-oat01-org-primary", "org@example.com")
  assert "sk-ant-oat01-henry@example.com-access" not in second.tokens

  # A user with no pool of their own resolves exactly what the org resolves.
  assert second.keys == resolve_anthropic_credentials(environ=environ).keys


def test_setup_token_enrolls_as_a_sibling_keyed_by_its_account(tmp_path: Path) -> None:
  """`claude setup-token` accounts join a pool without a refresh grant.

  Anthropic's profile endpoint refuses a setup token's scope, so the account
  key is derived from the token: two users who enroll the same account share
  one key, and one usage-limited account is therefore parked for both.
  """

  environ = {"USER_DATA_DIR": str(tmp_path)}
  token = "sk-ant-oat01-shared-subscription-token"
  first = enroll_anthropic_setup_token(
    token,
    store_path=resolve_anthropic_auth_store_path(environ=environ, user_scope=1),
  )
  second = enroll_anthropic_setup_token(
    token,
    store_path=resolve_anthropic_auth_store_path(environ=environ, user_scope=2),
  )

  assert first.identity == second.identity
  assert token not in first.identity
  assert first.refresh_token == ""
  assert first.needs_refresh(now=9_999_999_999.0) is False
  assert resolve_anthropic_credentials(environ=environ, user_scope=1).keys == (
    first.identity,
  )


def test_refresh_is_written_back_to_the_store_that_owns_the_account(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  """A user's expired account refreshes into the user's store, not the org's."""

  environ = {"USER_DATA_DIR": str(tmp_path)}
  user_store = resolve_anthropic_auth_store_path(environ=environ, user_scope=4)
  org_store = resolve_anthropic_auth_store_path(environ=environ)
  upsert_anthropic_oauth_record(user_store, _record("stale@example.com", expires_at=0.0))
  upsert_anthropic_oauth_record(org_store, _record("org@example.com"))
  monkeypatch.setattr(
    anthropic_oauth,
    "_post_token_endpoint",
    lambda url, body: {
      "access_token": "sk-ant-oat01-user-fresh",
      "refresh_token": "sk-ant-ort01-user-fresh",
      "expires_in": 28800,
    },
  )

  sources = resolve_anthropic_credentials(environ=environ, user_scope=4)
  pool = anthropic_oauth.AnthropicCredentialPool()
  assert select_anthropic_credential(sources, pool=pool) == "sk-ant-oat01-user-fresh"

  [stored] = load_anthropic_oauth_store(user_store)
  assert stored.access_token == "sk-ant-oat01-user-fresh"
  [org] = load_anthropic_oauth_store(org_store)
  assert org.access_token == "sk-ant-oat01-org@example.com-access"


def test_cli_enrolls_a_setup_token_from_stdin_into_one_users_pool(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  """`auth enroll anthropic --setup-token --user 1`, the production recipe.

  The token is never an argument and never echoed: it arrives on stdin and the
  command reports the account key and a masked token. It lands in that user's
  store, so the org pool other users are served is unchanged.
  """

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  monkeypatch.delenv("ANTHROPIC_AUTH_STORE_PATH", raising=False)
  token = "sk-ant-oat01-henry-personal-subscription-token"
  monkeypatch.setattr(agent_cli.sys, "stdin", io.StringIO(f"{token}\n"))
  stdout = io.StringIO()

  assert agent_cli.main(
    ["auth", "enroll", "anthropic", "--setup-token", "--user", "1"], stdout=stdout
  ) == 0

  printed = stdout.getvalue()
  assert token not in printed
  assert "sk-ant-oat01…oken" in printed
  [record] = load_anthropic_oauth_store(
    tmp_path / "users" / "1" / "anthropic" / "oauth.json"
  )
  assert record.access_token == token
  assert load_anthropic_oauth_store(tmp_path / "anthropic" / "oauth.json") == ()
  assert resolve_anthropic_credentials(
    environ={"USER_DATA_DIR": str(tmp_path)}, user_scope=1
  ).tokens == (token,)
