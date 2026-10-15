from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from threading import Event

import pytest

from agent_gateway import create_agent
from agent_gateway import cli as agent_cli
from agent_gateway.providers import anthropic_oauth
from agent_gateway.providers.anthropic_oauth import (
  ANTHROPIC_OAUTH_CLIENT_ID,
  ANTHROPIC_USER_SCOPE_FIELD,
  AnthropicOAuthError,
  AnthropicOAuthRecord,
  AnthropicSourceError,
  complete_anthropic_login,
  enroll_anthropic_setup_token,
  load_anthropic_oauth_store,
  parse_anthropic_callback,
  refresh_anthropic_oauth_record,
  register_anthropic_static_credential,
  remove_anthropic_oauth_record,
  resolve_anthropic_auth_store_path,
  resolve_anthropic_credentials,
  select_anthropic_credential_source,
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
  # Parking remains account-level for enrolled grants and token-level for env
  # members; durable source identity does not change pool cooldown semantics.
  assert sources.keys[-1] == "stored@example.com"
  env_credential = sources.by_token("sk-ant-oat01-existing-session")
  assert env_credential is not None
  assert env_credential.record is not None
  assert env_credential.record.source.startswith("env:ANTHROPIC_AUTH_TOKEN:")
  assert env_credential.store_path == path


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
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
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
  selected = select_anthropic_credential_source(sources, pool=pool)
  assert selected is not None
  assert selected.token == "sk-ant-oat01-bound-fresh"
  [stored] = load_anthropic_oauth_store(path)
  assert selected.record == stored
  assert selected.store_path == path
  assert selected.source_metadata() == stored.source_metadata()


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

  selected = select_anthropic_credential_source(sources, pool=pool)
  assert selected is not None
  assert selected.token == "sk-ant-oat01-healthy@example.com-access"
  stale, healthy = load_anthropic_oauth_store(path)
  # The record is marked, never deleted: re-enrolling is the operator's remedy.
  assert stale.identity == "stale@example.com"
  assert stale.last_error
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
  selected = select_anthropic_credential_source(sources, pool=pool)
  assert selected is not None
  assert selected.token == "sk-ant-oat01-user-fresh"
  assert selected.store_path == user_store

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


@pytest.mark.parametrize("legacy", [True, False])
def test_legacy_migration_persists_one_generation_across_reload(
  tmp_path: Path, legacy: bool,
) -> None:
  path = tmp_path / "oauth.json"
  original = _record("migrated@example.com")
  if legacy:
    payload = {"auth_token": original.access_token, "source": "claude-setup-token"}
    grant_material = original.access_token
  else:
    entry = original.to_payload()
    for name in ("generation_id", "grant_hash", "credential_kind"):
      entry.pop(name)
    payload = {"version": 2, "accounts": [entry]}
    grant_material = original.refresh_token
  path.write_text(json.dumps(payload), encoding="utf-8")
  path.chmod(0o644)

  [first] = load_anthropic_oauth_store(path)
  migrated_bytes = path.read_bytes()
  [second] = load_anthropic_oauth_store(path)
  assert first.source_metadata() == second.source_metadata()
  assert first.grant_hash == hashlib.sha256(grant_material.encode()).hexdigest()
  assert first.generation_id
  assert path.read_bytes() == migrated_bytes
  assert stat.S_IMODE(path.stat().st_mode) == 0o600
  recovered = resolve_anthropic_credentials(
    {"credential_source": first.source_metadata()},
    environ={"ANTHROPIC_AUTH_STORE_PATH": str(path)},
  ).credentials[0]
  assert recovered.token == original.access_token
  assert recovered.source_metadata() == first.source_metadata()


def test_exact_materialization_refreshes_material_without_changing_source_grant(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  path = tmp_path / "oauth.json"
  [original] = upsert_anthropic_oauth_record(
    path, _record("grant@example.com", expires_at=0.0),
  )
  metadata = original.source_metadata()
  env = {"ANTHROPIC_AUTH_STORE_PATH": str(path)}
  monkeypatch.setattr(
    anthropic_oauth, "_post_token_endpoint",
    lambda url, body: {
      "access_token": "fixture-rotated-access",
      "refresh_token": "fixture-rotated-refresh",
      "expires_in": 28800,
    },
  )
  sources = resolve_anthropic_credentials({"credential_source": metadata}, environ=env)
  materialized = select_anthropic_credential_source(
    sources, pool=anthropic_oauth.AnthropicCredentialPool(), now=1000.0,
  )
  assert materialized is not None
  assert materialized.token == "fixture-rotated-access"
  assert materialized.source_metadata() == metadata
  assert set(metadata) == {"account_identity", "grant_hash", "generation_id", "generation_ref"}
  assert original.access_token not in metadata["generation_ref"]
  assert str(path) not in metadata["generation_ref"]
  assert metadata["grant_hash"] == hashlib.sha256(original.refresh_token.encode()).hexdigest()
  recovered = resolve_anthropic_credentials(
    {"credential_source": metadata}, environ=env,
  ).credentials[0]
  assert recovered == materialized
  [stored] = load_anthropic_oauth_store(path)
  assert stored.refresh_token == "fixture-rotated-refresh"
  assert stored.source_metadata() == metadata
  assert original.refresh_token not in path.read_text()


def test_same_account_reconsent_supersedes_original_generation_in_its_pool_slot(
  tmp_path: Path,
) -> None:
  path = tmp_path / "oauth.json"
  env = {"ANTHROPIC_AUTH_STORE_PATH": str(path)}

  def consent(access: str, refresh: str) -> AnthropicOAuthRecord:
    request = start_anthropic_login()
    return complete_anthropic_login(
      request, f"fixture-code#{request.state}", store_path=path, now=1000.0,
      post=lambda url, body: {
        "access_token": access,
        "refresh_token": refresh,
        "expires_in": 28800,
        "account": {"email_address": "same@example.com"},
      },
    )

  original = consent("fixture-first-access", "fixture-first-grant")
  upsert_anthropic_oauth_record(path, _record("sibling@example.com"))
  replacement = consent("fixture-next-access", "fixture-next-grant")

  assert replacement.identity == original.identity
  assert replacement.grant_hash != original.grant_hash
  assert replacement.generation_id != original.generation_id
  active = load_anthropic_oauth_store(path)
  assert [record.identity for record in active] == ["same@example.com", "sibling@example.com"]
  assert active[0] == replacement
  with pytest.raises(AnthropicSourceError) as missing:
    resolve_anthropic_credentials({"credential_source": original.source_metadata()}, environ=env)
  assert missing.value.code == "credential_unavailable"
  assert missing.value.reason == "superseded"
  assert missing.value.next_call
  assert resolve_anthropic_credentials(
    {"credential_source": replacement.source_metadata()}, environ=env,
  ).credentials[0].token == replacement.access_token
  assert original.access_token not in path.read_text()
  assert original.refresh_token not in path.read_text()


@pytest.mark.parametrize("source", ["env:ANTHROPIC_API_KEY", "request:anthropic:api_key"])
def test_static_api_key_reobservation_and_replacement_have_exact_source_generations(
  tmp_path: Path, source: str,
) -> None:
  env = {"USER_DATA_DIR": str(tmp_path), "ANTHROPIC_API_KEY": "fixture-api-key-first"}
  original = register_anthropic_static_credential(
    env["ANTHROPIC_API_KEY"], source=source, environ=env, user_scope=7,
  )
  assert original is not None and original.record is not None
  path = resolve_anthropic_auth_store_path(
    environ=env, user_scope="" if source.startswith("env:") else 7,
  )
  assert original.store_path == path
  assert original.account_identity == "api-key-" + hashlib.sha256(original.token.encode()).hexdigest()[:16]
  assert original.grant_hash == hashlib.sha256(original.token.encode()).hexdigest()
  repeated = register_anthropic_static_credential(
    original.token, source=source, environ=env, user_scope=7,
  )
  assert repeated is not None
  assert repeated.source_metadata() == original.source_metadata()
  assert load_anthropic_oauth_store(path) == ()
  assert resolve_anthropic_credentials(environ=env, user_scope=7).credentials == ()

  env["ANTHROPIC_API_KEY"] = "fixture-api-key-replacement"
  replacement = register_anthropic_static_credential(
    env["ANTHROPIC_API_KEY"], source=source, environ=env, user_scope=7,
  )
  assert replacement is not None
  assert replacement.generation_id != original.generation_id
  assert load_anthropic_oauth_store(path) == ()
  assert original.token not in path.read_text()
  assert stat.S_IMODE(path.stat().st_mode) == 0o600
  if source.startswith("env:"):
    assert not resolve_anthropic_auth_store_path(environ=env, user_scope=7).exists()


def test_empty_static_source_does_not_invent_a_generation_and_removal_discards_material(
  tmp_path: Path,
) -> None:
  env = {"USER_DATA_DIR": str(tmp_path)}
  assert register_anthropic_static_credential("", environ=env) is None
  path = resolve_anthropic_auth_store_path(environ=env)
  assert not path.exists()
  key = register_anthropic_static_credential(
    "fixture-request-key", source="request:anthropic:api_key", environ=env, user_scope=1,
  )
  assert key is not None
  assert register_anthropic_static_credential(
    "", source="request:anthropic:api_key", environ=env, user_scope=1,
  ) is None
  assert key.store_path is not None
  saved = json.loads(key.store_path.read_text())
  assert saved["static_credentials"] == []
  assert key.token not in key.store_path.read_text()


def test_exact_generation_lookup_never_crosses_user_custody_or_matches_same_account(
  tmp_path: Path,
) -> None:
  env = {"USER_DATA_DIR": str(tmp_path)}
  user_path = resolve_anthropic_auth_store_path(environ=env, user_scope=1)
  org_path = resolve_anthropic_auth_store_path(environ=env)
  [private] = upsert_anthropic_oauth_record(user_path, _record("same@example.com"))
  [shared] = upsert_anthropic_oauth_record(
    org_path, replace(_record("same@example.com"), access_token="fixture-org-access"),
  )
  assert resolve_anthropic_credentials(
    {"credential_source": private.source_metadata()}, environ=env, user_scope=1,
  ).credentials[0].token == private.access_token
  with pytest.raises(AnthropicSourceError) as missing:
    resolve_anthropic_credentials(
      {"credential_source": private.source_metadata()}, environ=env, user_scope=2,
    )
  assert missing.value.reason == "missing"
  assert resolve_anthropic_credentials(
    {"credential_source": shared.source_metadata()}, environ=env, user_scope=2,
  ).credentials[0].token == shared.access_token
  with pytest.raises(AnthropicSourceError) as unknown:
    resolve_anthropic_credentials(
      {"credential_source": {"generation_ref": "fixture-secret-ref-not-enrolled"}},
      environ=env, user_scope=1,
    )
  assert "fixture-secret-ref-not-enrolled" not in str(unknown.value)


def test_environment_array_sources_keep_membership_identity_across_reordering_and_removal(
  tmp_path: Path,
) -> None:
  env = {
    "USER_DATA_DIR": str(tmp_path),
    "ANTHROPIC_AUTH_TOKENS": json.dumps(["fixture-member-a", "fixture-member-b"]),
  }
  original = resolve_anthropic_credentials(environ=env, user_scope=1)
  first, second = original.credentials
  assert first.store_path == second.store_path == original.store_path
  assert load_anthropic_oauth_store(original.store_path) == ()
  assert original.user_store_path is not None and not original.user_store_path.exists()
  env["ANTHROPIC_AUTH_TOKENS"] = json.dumps(["fixture-member-b", "fixture-member-a"])
  reordered = resolve_anthropic_credentials(environ=env, user_scope=2)
  assert reordered.tokens == ("fixture-member-b", "fixture-member-a")
  assert [credential.generation_ref for credential in reordered.credentials] == [
    second.generation_ref, first.generation_ref,
  ]
  env["ANTHROPIC_AUTH_TOKENS"] = json.dumps(["fixture-member-b", "fixture-member-c"])
  updated = resolve_anthropic_credentials(environ=env, user_scope=2)
  assert updated.tokens == ("fixture-member-b", "fixture-member-c")
  assert updated.credentials[0].generation_ref == second.generation_ref
  assert updated.credentials[1].generation_ref not in (first.generation_ref, second.generation_ref)
  with pytest.raises(AnthropicSourceError) as removed:
    resolve_anthropic_credentials(
      {"credential_source": first.source_metadata()}, environ=env, user_scope=2,
    )
  assert removed.value.reason == "removed"
  assert resolve_anthropic_credentials(
    {"credential_source": second.source_metadata()}, environ=env, user_scope=2,
  ).credentials[0].token == second.token
  assert first.token not in original.store_path.read_text()


def test_environment_singleton_replacement_preserves_other_env_source_members(
  tmp_path: Path,
) -> None:
  env = {
    "USER_DATA_DIR": str(tmp_path),
    "ANTHROPIC_AUTH_TOKEN": "fixture-primary-env-token",
    "CLAUDE_CODE_OAUTH_TOKEN": "fixture-other-env-token",
  }
  first, other = resolve_anthropic_credentials(environ=env).credentials
  env["ANTHROPIC_AUTH_TOKEN"] = "fixture-replacement-env-token"
  replacement, retained = resolve_anthropic_credentials(environ=env).credentials
  assert retained.generation_ref == other.generation_ref
  assert replacement.generation_ref != first.generation_ref
  with pytest.raises(AnthropicSourceError) as missing:
    resolve_anthropic_credentials({"credential_source": first.source_metadata()}, environ=env)
  assert missing.value.reason == "superseded"
  assert resolve_anthropic_credentials(
    {"credential_source": retained.source_metadata()}, environ=env,
  ).credentials[0].token == other.token


def test_captured_refresh_record_cannot_overwrite_new_consent(
  tmp_path: Path,
) -> None:
  path = tmp_path / "oauth.json"
  [original] = upsert_anthropic_oauth_record(path, _record("same@example.com", expires_at=0.0))
  [replacement] = upsert_anthropic_oauth_record(
    path, replace(_record("same@example.com"), access_token="fixture-new-consent", refresh_token="fixture-new-grant"),
  )
  posts: list[object] = []
  with pytest.raises(AnthropicSourceError) as missing:
    refresh_anthropic_oauth_record(
      original, store_path=path, now=1000.0,
      post=lambda url, body: posts.append(body) or {},
    )
  assert missing.value.reason == "superseded"
  assert posts == []
  assert load_anthropic_oauth_store(path) == (replacement,)


def test_concurrent_refreshers_never_reuse_a_captured_rotating_refresh_token(
  tmp_path: Path,
) -> None:
  path = tmp_path / "oauth.json"
  [original] = upsert_anthropic_oauth_record(path, _record("same@example.com", expires_at=0.0))
  entered = Event()
  release = Event()
  second_started = Event()
  posted: list[str] = []

  def post(url, body):
    posted.append(body["refresh_token"])
    entered.set()
    assert release.wait(5)
    return {"access_token": "fixture-fresh-access", "refresh_token": "fixture-fresh-refresh", "expires_in": 28800}

  def second_refresh():
    second_started.set()
    return refresh_anthropic_oauth_record(original, store_path=path, now=1000.0, post=post)

  with ThreadPoolExecutor(max_workers=2) as executor:
    first = executor.submit(refresh_anthropic_oauth_record, original, store_path=path, now=1000.0, post=post)
    try:
      assert entered.wait(5)
      second = executor.submit(second_refresh)
      assert second_started.wait(5)
    finally:
      release.set()
    first_result, second_result = first.result(5), second.result(5)
  assert posted == [original.refresh_token]
  assert first_result == second_result
  assert first_result.source_metadata() == original.source_metadata()
  assert load_anthropic_oauth_store(path) == (first_result,)


@pytest.mark.parametrize("mutation", ["consent", "remove"])
def test_consent_and_removal_serialize_with_refresh_and_cannot_be_undone(
  tmp_path: Path, mutation: str,
) -> None:
  path = tmp_path / "oauth.json"
  env = {"ANTHROPIC_AUTH_STORE_PATH": str(path)}
  [original] = upsert_anthropic_oauth_record(path, _record("same@example.com", expires_at=0.0))
  entered = Event()
  release = Event()
  mutation_started = Event()

  def post(url, body):
    entered.set()
    assert release.wait(5)
    return {"access_token": "fixture-refreshed-access", "refresh_token": "fixture-refreshed-refresh", "expires_in": 28800}

  def mutate():
    mutation_started.set()
    if mutation == "remove":
      return remove_anthropic_oauth_record(path, original.identity)
    return upsert_anthropic_oauth_record(
      path, replace(_record(original.identity), access_token="fixture-consent-access", refresh_token="fixture-consent-grant"),
    )

  with ThreadPoolExecutor(max_workers=2) as executor:
    refresh = executor.submit(refresh_anthropic_oauth_record, original, store_path=path, now=1000.0, post=post)
    try:
      assert entered.wait(5)
      mutator = executor.submit(mutate)
      assert mutation_started.wait(5)
    finally:
      release.set()
    refresh.result(5)
    mutator.result(5)
  active = load_anthropic_oauth_store(path)
  if mutation == "remove":
    assert active == ()
  else:
    assert len(active) == 1
    assert active[0].access_token == "fixture-consent-access"
    assert active[0].generation_id != original.generation_id
  with pytest.raises(AnthropicSourceError) as missing:
    resolve_anthropic_credentials({"credential_source": original.source_metadata()}, environ=env)
  assert missing.value.reason == ("removed" if mutation == "remove" else "superseded")
  assert "fixture-refreshed-access" not in path.read_text()
  assert "fixture-refreshed-refresh" not in path.read_text()


def test_source_repr_and_refresh_failure_never_expose_credential_material(
  tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
  path = tmp_path / "oauth.json"
  [record] = upsert_anthropic_oauth_record(path, _record("private@example.com", expires_at=0.0))
  sources = resolve_anthropic_credentials(environ={"ANTHROPIC_AUTH_STORE_PATH": str(path)})
  for material in (record.access_token, record.refresh_token):
    assert material not in repr(record)
    assert material not in repr(sources)

  def failure(url, body):
    raise AnthropicOAuthError(f"fixture endpoint echoed {record.refresh_token}")

  with pytest.raises(AnthropicOAuthError) as failed:
    refresh_anthropic_oauth_record(record, store_path=path, now=1000.0, post=failure)
  monkeypatch.setattr(anthropic_oauth, "_post_token_endpoint", failure)
  caplog.set_level("INFO", logger=anthropic_oauth.__name__)
  selected = select_anthropic_credential_source(sources, pool=anthropic_oauth.AnthropicCredentialPool(), now=1000.0)
  assert selected is not None
  assert selected.source_metadata() == record.source_metadata()
  assert caplog.records
  [stored] = load_anthropic_oauth_store(path)
  for material in (record.access_token, record.refresh_token):
    assert material not in str(failed.value)
    assert material not in stored.last_error
    assert material not in caplog.text


def test_empty_pool_selection_returns_no_source_or_invented_generation(tmp_path: Path) -> None:
  env = {"USER_DATA_DIR": str(tmp_path)}
  sources = resolve_anthropic_credentials(environ=env, user_scope=3)
  assert select_anthropic_credential_source(sources, pool=anthropic_oauth.AnthropicCredentialPool()) is None
  assert not sources.store_path.exists()
  assert sources.user_store_path is not None and not sources.user_store_path.exists()


def test_reconsent_with_identical_returned_material_still_creates_a_new_generation(
  tmp_path: Path,
) -> None:
  path = tmp_path / "oauth.json"
  request = start_anthropic_login()

  def login():
    return complete_anthropic_login(
      request, f"fixture-code#{request.state}", store_path=path, now=1000.0,
      post=lambda url, body: {
        "access_token": "fixture-consent-access",
        "refresh_token": "fixture-consent-refresh",
        "expires_in": 28800,
        "account": {"email_address": "same@example.com"},
      },
    )

  original, replacement = login(), login()
  assert original.grant_hash == replacement.grant_hash
  assert original.generation_id != replacement.generation_id
  assert load_anthropic_oauth_store(path) == (replacement,)
  with pytest.raises(AnthropicSourceError) as missing:
    resolve_anthropic_credentials(
      {"credential_source": original.source_metadata()},
      environ={"ANTHROPIC_AUTH_STORE_PATH": str(path)},
    )
  assert missing.value.reason == "superseded"


@pytest.mark.parametrize("with_metadata", [False, True])
def test_bound_environment_config_preserves_original_org_source_without_user_clone(
  tmp_path: Path, with_metadata: bool,
) -> None:
  env = {"USER_DATA_DIR": str(tmp_path), "ANTHROPIC_AUTH_TOKEN": "fixture-org-token"}
  original_sources = resolve_anthropic_credentials(environ=env, user_scope=7)
  original = select_anthropic_credential_source(
    original_sources, pool=anthropic_oauth.AnthropicCredentialPool(),
  )
  assert original is not None
  cfg: dict[str, object] = {"auth_token": original.token, ANTHROPIC_USER_SCOPE_FIELD: "7"}
  if with_metadata:
    cfg["credential_source"] = original.source_metadata()

  repeated_sources = resolve_anthropic_credentials(cfg, environ=env)
  repeated = select_anthropic_credential_source(
    repeated_sources, pool=anthropic_oauth.AnthropicCredentialPool(),
  )
  assert repeated is not None
  assert repeated.generation_ref == original.generation_ref
  assert repeated.store_path == original_sources.store_path
  assert repeated.record is not None and repeated.record.source.startswith("env:ANTHROPIC_AUTH_TOKEN:")
  assert anthropic_oauth.ensure_fresh_anthropic_credential(original.token, config=cfg, environ=env) == original.token
  assert original_sources.user_store_path is not None
  assert not original_sources.user_store_path.exists()
  saved = json.loads(original_sources.store_path.read_text())
  assert saved["static_credentials"][0]["source"].startswith("env:ANTHROPIC_AUTH_TOKEN:")

  env["ANTHROPIC_AUTH_TOKEN"] = ""
  with pytest.raises(AnthropicSourceError) as removed:
    resolve_anthropic_credentials(cfg, environ=env)
  assert removed.value.reason == "removed"
  with pytest.raises(AnthropicSourceError):
    resolve_anthropic_credentials(
      {"credential_source": original.source_metadata()}, environ=env, user_scope=7,
    )
  assert not original_sources.user_store_path.exists()
  if with_metadata:
    # An exact `credential_source` lookup only reads: the emptied source is
    # reported `removed` and the store is left byte-for-byte unchanged.
    assert original.token in original_sources.store_path.read_text()
  else:
    # A pool resolve reconciles the observed empty source and retires it.
    assert original.token not in original_sources.store_path.read_text()


def test_bound_source_metadata_retains_original_grant_after_access_token_refresh(
  tmp_path: Path,
) -> None:
  env = {"USER_DATA_DIR": str(tmp_path)}
  user_path = resolve_anthropic_auth_store_path(environ=env, user_scope=7)
  [original] = upsert_anthropic_oauth_record(
    user_path, _record("same@example.com", expires_at=0.0),
  )
  cfg: dict[str, object] = {
    "auth_token": original.access_token,
    ANTHROPIC_USER_SCOPE_FIELD: "7",
    "credential_source": original.source_metadata(),
  }
  refreshed = refresh_anthropic_oauth_record(
    original, store_path=user_path, now=1000.0,
    post=lambda url, body: {
      "access_token": "fixture-grant-refreshed-access",
      "refresh_token": "fixture-grant-refreshed-refresh",
      "expires_in": 28800,
    },
  )
  resolved = resolve_anthropic_credentials(cfg, environ=env)
  assert len(resolved.credentials) == 1
  assert resolved.credentials[0].token == refreshed.access_token
  assert resolved.credentials[0].source_metadata() == original.source_metadata()
  assert anthropic_oauth.ensure_fresh_anthropic_credential(
    original.access_token, config=cfg, environ=env, now=1000.0,
  ) == refreshed.access_token
  saved = json.loads(user_path.read_text())
  assert saved["static_credentials"] == []

  [replacement] = upsert_anthropic_oauth_record(
    user_path,
    replace(_record(original.identity), access_token="fixture-new-consent", refresh_token="fixture-new-consent-grant"),
  )
  # Even bytes matching the current successor cannot substitute for known X.
  cfg["auth_token"] = replacement.access_token
  with pytest.raises(AnthropicSourceError) as superseded:
    resolve_anthropic_credentials(cfg, environ=env)
  assert superseded.value.reason == "superseded"
  with pytest.raises(AnthropicSourceError):
    anthropic_oauth.ensure_fresh_anthropic_credential(
      replacement.access_token, config=cfg, environ=env, now=1000.0,
    )
  assert load_anthropic_oauth_store(user_path) == (replacement,)
  assert json.loads(user_path.read_text())["static_credentials"] == []


@pytest.mark.parametrize("name", [
  "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_AUTH_TOKENS",
  "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY",
])
def test_process_without_source_environment_cannot_retire_durable_generation(
  tmp_path: Path, name: str,
) -> None:
  token = "fixture-authoritative-material"
  authoritative_env = {
    "USER_DATA_DIR": str(tmp_path),
    name: json.dumps([token]) if name == "ANTHROPIC_AUTH_TOKENS" else token,
  }
  if name == "ANTHROPIC_API_KEY":
    original = register_anthropic_static_credential(token, environ=authoritative_env)
  else:
    [original] = resolve_anthropic_credentials(environ=authoritative_env).credentials
  assert original is not None and original.store_path is not None
  saved_bytes = original.store_path.read_bytes()
  unobserving_env = {"USER_DATA_DIR": str(tmp_path)}

  assert resolve_anthropic_credentials(environ=unobserving_env).credentials == ()
  assert original.store_path.read_bytes() == saved_bytes
  if name != "ANTHROPIC_API_KEY":
    for with_metadata in (False, True):
      cfg: dict[str, object] = {"auth_token": token, ANTHROPIC_USER_SCOPE_FIELD: "7"}
      if with_metadata:
        cfg["credential_source"] = original.source_metadata()
      with pytest.raises(AnthropicSourceError) as unavailable_bound:
        resolve_anthropic_credentials(cfg, environ=unobserving_env)
      assert unavailable_bound.value.reason == "unobservable"
      assert original.store_path.read_bytes() == saved_bytes
      assert not resolve_anthropic_auth_store_path(environ=unobserving_env, user_scope=7).exists()
    [recovered] = resolve_anthropic_credentials(environ=authoritative_env).credentials
  else:
    recovered = register_anthropic_static_credential(token, environ=authoritative_env)
  assert recovered is not None
  assert recovered.source_metadata() == original.source_metadata()
  assert original.store_path.read_bytes() == saved_bytes


def test_partial_environment_retires_only_observed_empty_source(
  tmp_path: Path,
) -> None:
  env = {
    "USER_DATA_DIR": str(tmp_path),
    "ANTHROPIC_AUTH_TOKEN": "fixture-first-visible-token",
    "CLAUDE_CODE_OAUTH_TOKEN": "fixture-second-visible-token",
  }
  first, second = resolve_anthropic_credentials(environ=env).credentials
  partial_env = {"USER_DATA_DIR": str(tmp_path), "ANTHROPIC_AUTH_TOKEN": ""}
  assert resolve_anthropic_credentials(environ=partial_env).credentials == ()
  assert second.store_path is not None
  assert first.token not in second.store_path.read_text()
  with pytest.raises(AnthropicSourceError) as removed:
    resolve_anthropic_credentials({"credential_source": first.source_metadata()}, environ=partial_env)
  assert removed.value.reason == "removed"
  with pytest.raises(AnthropicSourceError) as unobservable:
    resolve_anthropic_credentials({"credential_source": second.source_metadata()}, environ=partial_env)
  assert unobservable.value.reason == "unobservable"
  assert resolve_anthropic_credentials(
    {"credential_source": second.source_metadata()}, environ=env,
  ).credentials[0].token == second.token



@pytest.mark.parametrize("credential_kind", ["api_key", "auth_token"])
def test_generation_auth_config_materializes_exact_source_and_rejects_retirement(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, credential_kind: str,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  frozen = register_anthropic_static_credential(
    "frozen-material", credential_kind=credential_kind,
    source="request:test:frozen", user_scope=7,
  )
  assert frozen is not None
  register_anthropic_static_credential(
    "other-material", credential_kind=credential_kind,
    source="request:test:other", user_scope=7,
  )
  config = anthropic_oauth.anthropic_generation_auth_config(
    frozen.generation_ref, user_scope=7,
    base_config={"max_tokens": 123, "api_key": "wrong", "auth_token": "wrong"},
  )
  assert config["max_tokens"] == 123
  assert config["credential_source"] == frozen.source_metadata()
  assert config["auth_user_scope"] == "7"
  assert config["api_key"] == ("frozen-material" if credential_kind == "api_key" else "")
  assert config["auth_token"] == ("frozen-material" if credential_kind == "auth_token" else "")
  register_anthropic_static_credential(
    "replacement", credential_kind=credential_kind,
    source="request:test:frozen", user_scope=7,
  )
  with pytest.raises(AnthropicSourceError):
    anthropic_oauth.anthropic_generation_auth_config(
      frozen.generation_ref, user_scope=7, base_config=config,
    )


@pytest.mark.parametrize("credential_kind", ["api_key", "auth_token"])
def test_request_enrollment_without_material_preserves_existing_generation(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, credential_kind: str,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  existing = register_anthropic_static_credential(
    "existing-material", credential_kind=credential_kind,
    source=f"request:anthropic:{credential_kind}", user_scope=7,
  )
  assert existing is not None
  path = resolve_anthropic_auth_store_path(user_scope=7)
  before = path.read_bytes()
  assert anthropic_oauth.enroll_request_anthropic_credential(
    "  ", credential_kind=credential_kind, user_scope=7,
  ) is None
  assert path.read_bytes() == before
  config = anthropic_oauth.anthropic_generation_auth_config(
    existing.generation_ref, user_scope=7, base_config={},
  )
  assert config["credential_source"] == existing.source_metadata()


@pytest.mark.parametrize("account_scope", [7, None])
def test_request_enrollment_retains_oauth_account_refresh_authority(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, account_scope: int | None,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  path = resolve_anthropic_auth_store_path(user_scope=account_scope)
  record = _record("request-account", expires_at=900.0)
  upsert_anthropic_oauth_record(path, record)
  owned = resolve_anthropic_credentials(user_scope=7).by_token(record.access_token)
  assert owned is not None and owned.record is not None
  entered = anthropic_oauth.enroll_request_anthropic_credential(
    record.access_token, credential_kind="auth_token", user_scope=7,
  )
  assert entered is not None and entered.record is not None
  assert entered.generation_ref == owned.generation_ref
  assert entered.record.refresh_token == record.refresh_token
  posts = []

  def post(url, body):
    posts.append(body)
    return {"access_token": "fresh-bearer", "expires_in": 3600}

  monkeypatch.setattr(anthropic_oauth, "_post_token_endpoint", post)
  config = anthropic_oauth.anthropic_generation_auth_config(
    entered.generation_ref, user_scope=7, base_config={},
  )
  assert anthropic_oauth.ensure_fresh_anthropic_credential(
    entered.token, config=config, now=1000.0,
  ) == "fresh-bearer"
  assert len(posts) == 1
  assert posts[0]["refresh_token"] == record.refresh_token


@pytest.mark.parametrize("user_scope", [None, "7"])
def test_bare_bound_tokens_of_different_requests_never_retire_each_other(
  tmp_path: Path, user_scope: str | None,
) -> None:
  env = {"USER_DATA_DIR": str(tmp_path)}
  generations: list[str] = []
  for token in ("fixture-user-a-token", "fixture-user-b-token", "fixture-user-a-token"):
    cfg: dict[str, object] = {"auth_mode": "oauth", "auth_token": token}
    if user_scope is not None:
      cfg[ANTHROPIC_USER_SCOPE_FIELD] = user_scope
    bound = resolve_anthropic_credentials(cfg, environ=env).credentials[0]
    assert bound.token == token
    generations.append(bound.generation_ref)
    assert anthropic_oauth.ensure_fresh_anthropic_credential(token, config=cfg, environ=env) == token
  first_a, b, second_a = generations
  assert first_a == second_a and first_a != b
  path = resolve_anthropic_auth_store_path(environ=env, user_scope=user_scope)
  saved = json.loads(path.read_text())
  assert saved["retirements"] == {}
  assert saved["retired_auth_token_grants"] == {}
  assert len(saved["static_credentials"]) == 2
  for ref in (first_a, b):
    resolved = resolve_anthropic_credentials(
      {"credential_source": {"generation_ref": ref}}, environ=env, user_scope=user_scope,
    ).credentials[0]
    assert resolved.generation_ref == ref


@pytest.mark.parametrize("credential_kind", ["api_key", "auth_token"])
@pytest.mark.parametrize("user_scope", [None, 7])
def test_request_enrollment_of_different_material_keeps_every_generation(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
  credential_kind: str, user_scope: int | None,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  first, second, again = (
    anthropic_oauth.enroll_request_anthropic_credential(
      material, credential_kind=credential_kind, user_scope=user_scope,
    )
    for material in ("first-session-material", "second-session-material", "first-session-material")
  )
  assert first is not None and second is not None and again is not None
  assert again.generation_ref == first.generation_ref != second.generation_ref
  for enrolled in (first, second):
    config = anthropic_oauth.anthropic_generation_auth_config(
      enrolled.generation_ref, user_scope=user_scope, base_config={},
    )
    assert config["credential_source"] == enrolled.source_metadata()


def test_exact_lookup_and_read_paths_never_write_a_read_only_store(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Regression a: a read path opens the store read-only, never a raw OSError.

  A store directory and lock file this process cannot open for write is exactly
  a read-only deployment. The exact lookup reads through it; the pool writer
  raises the owner's typed store error. The store bytes never change.
  """
  path = tmp_path / "anthropic" / "oauth.json"
  [record] = upsert_anthropic_oauth_record(path, _record("readonly@example.com"))
  saved = path.read_bytes()
  lock_path = path.with_name(f".{path.name}.lock")
  # Deny every write to the store's directory and lock file while leaving the
  # store itself readable: `os.open(..., O_WRONLY)` and `mkdir` both fail.
  real_open = os.open

  def guarded_open(target, flags, *args, **kwargs):
    target_path = Path(target)
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT) and (
      target_path == lock_path or target_path == path
      or target_path.parent == path.parent
    ):
      raise PermissionError(13, "Permission denied", str(target))
    return real_open(target, flags, *args, **kwargs)

  real_mkdir = Path.mkdir

  def guarded_mkdir(self, *args, **kwargs):
    if self == path.parent:
      raise PermissionError(13, "Permission denied", str(self))
    return real_mkdir(self, *args, **kwargs)

  monkeypatch.setattr(os, "open", guarded_open)
  monkeypatch.setattr(Path, "mkdir", guarded_mkdir)

  # The exact lookup only reads: it succeeds and leaves the store unchanged.
  resolved = resolve_anthropic_credentials(
    {"credential_source": record.source_metadata()},
    environ={"ANTHROPIC_AUTH_STORE_PATH": str(path)},
  )
  assert resolved.credentials[0].generation_ref == record.generation_ref
  assert path.read_bytes() == saved
  # A pool resolve reconciles env and therefore writes: an unwritable store is
  # the owner's typed error, never a raw PermissionError.
  from agent_gateway.providers.anthropic_oauth import AnthropicStoreError

  with pytest.raises(AnthropicStoreError) as unreadable:
    resolve_anthropic_credentials(
      environ={"ANTHROPIC_AUTH_STORE_PATH": str(path), "ANTHROPIC_AUTH_TOKEN": "fixture"},
    )
  assert unreadable.value.code == "store_unreadable"


def test_exact_lookup_does_not_sync_environment_or_mutate_the_store(
  tmp_path: Path,
) -> None:
  """Regression b: an exact generation resolve reports the bound generation, no sync."""
  path = tmp_path / "anthropic" / "oauth.json"
  [record] = upsert_anthropic_oauth_record(path, _record("bound@example.com"))
  saved = path.read_bytes()
  cfg = {"auth_token": record.access_token, "credential_source": record.source_metadata()}
  env = {"ANTHROPIC_AUTH_STORE_PATH": str(path), "ANTHROPIC_AUTH_TOKEN": "fixture-env-token"}
  resolved = resolve_anthropic_credentials(cfg, environ=env)
  assert len(resolved.credentials) == 1
  assert resolved.credentials[0].token == record.access_token
  assert resolved.credentials[0].generation_ref == record.generation_ref
  # No env sync happened: the store bytes are exactly as enrolled.
  assert path.read_bytes() == saved


def test_two_processes_with_different_env_tokens_do_not_retire_each_other(
  tmp_path: Path,
) -> None:
  """Regression c: differing env tokens sharing one org store stay resolvable."""
  path = tmp_path / "anthropic" / "oauth.json"
  base = {"ANTHROPIC_AUTH_STORE_PATH": str(path)}
  first = resolve_anthropic_credentials(
    environ={**base, "ANTHROPIC_AUTH_TOKEN": "fixture-token-a"},
  ).by_token("fixture-token-a")
  assert first is not None
  second = resolve_anthropic_credentials(
    environ={**base, "ANTHROPIC_AUTH_TOKEN": "fixture-token-b"},
  ).by_token("fixture-token-b")
  assert second is not None
  # Both generations remain resolvable by their own exact source.
  for credential, env_token in ((first, "fixture-token-a"), (second, "fixture-token-b")):
    again = resolve_anthropic_credentials(
      {"credential_source": credential.source_metadata()},
      environ={**base, "ANTHROPIC_AUTH_TOKEN": env_token},
    )
    assert again.credentials[0].generation_ref == credential.generation_ref
