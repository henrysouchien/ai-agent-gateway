from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_gateway import cli as agent_cli
from agent_gateway.login_flows import get_login_flow
from agent_gateway.user_provider_keys_client import EnrolmentRefused, RiskUserKeyClient
from hank_contracts.user_provider_keys import EnrolmentOk, EnrolmentRefusal, KeyView, Provider


FINGERPRINT = "abcdef012345"
SECRET = "sk-test-user-api-key-not-for-output"
VALIDATED = "2026-11-03T10:00:00Z"


def key_view(*, suspended: bool = False) -> KeyView:
  return KeyView(
    fingerprint=FINGERPRINT, label="research", status="suspended" if suspended else "active",
    stop_kind="out_of_credit" if suspended else None, suspended_until=None,
    created_at=VALIDATED, last_validated_at=VALIDATED,
  )


@pytest.fixture
def door(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[tuple[MagicMock, MagicMock]]:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  monkeypatch.delenv("HANK_USER_RUNTIME_CREDENTIALS_FILE", raising=False)
  monkeypatch.setenv("GATEWAY_USER_KEYS", json.dumps([
    {"slug": "analyst", "email": "analyst@example.com", "risk_user_id": 731},
  ]))
  monkeypatch.setenv("RISK_MODULE_RESOLVER_URL", "https://risk.invalid/api/internal/resolve-credential")
  monkeypatch.setenv("GATEWAY_RESOLVER_HMAC_KEY", "test-hmac-not-a-real-secret")
  client = MagicMock(spec=RiskUserKeyClient)
  with patch("agent_gateway.login_flows.RiskUserKeyClient", return_value=client) as factory:
    yield client, factory
  assert list(tmp_path.rglob("*")) == [], "User API keys must never create gateway files"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("user", ["analyst", "ANALYST@example.com"])
def test_login_prompts_secret_and_binds_roster_identity(
  door: tuple[MagicMock, MagicMock], provider: Provider, user: str,
  caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
  client, factory = door
  client.add_key.return_value = EnrolmentOk(731, provider, (key_view(),))
  argv = ["auth", "login", provider, "--api-key", "--user", user, "--label", "research"]
  workspace_id = "workspace-test" if provider == "anthropic" else None
  if workspace_id is not None:
    argv += ["--workspace-id", workspace_id]
  stdout, stderr = StringIO(), StringIO()
  with patch("getpass.getpass", return_value=SECRET) as prompt:
    assert agent_cli.main(argv, stdout=stdout, stderr=stderr) == 0
  prompt.assert_called_once()
  client.add_key.assert_called_once_with(
    risk_user_id=731, provider=provider, api_key=SECRET, label="research", workspace_id=workspace_id,
  )
  factory.assert_called_once_with(
    resolve_url="https://risk.invalid/api/internal/resolve-credential",
    hmac_key="test-hmac-not-a-real-secret",
  )
  assert FINGERPRINT in stdout.getvalue()
  captured = capsys.readouterr()
  assert SECRET not in " ".join(argv)
  assert SECRET not in stdout.getvalue() + stderr.getvalue() + caplog.text + captured.out + captured.err


def test_login_uses_default_label(door: tuple[MagicMock, MagicMock]) -> None:
  client, _ = door
  client.add_key.return_value = EnrolmentOk(731, "openai", (key_view(),))
  with patch("getpass.getpass", return_value=SECRET):
    assert agent_cli.main(
      ["auth", "login", "openai", "--api-key", "--user", "analyst"], stdout=StringIO(),
    ) == 0
  assert client.add_key.call_args.kwargs["label"] == "cli"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("command", ["login", "status", "logout"])
@pytest.mark.parametrize("user", [None, "unknown", "731"])
def test_unresolved_identity_never_prompts_or_requests(
  door: tuple[MagicMock, MagicMock], provider: str, command: str, user: str | None,
) -> None:
  client, factory = door
  argv = ["auth", command, provider, "--api-key"]
  if user is not None:
    argv += ["--user", user]
  if command == "logout":
    argv += ["--key", FINGERPRINT]
  stdout, stderr = StringIO(), StringIO()
  with patch("getpass.getpass") as prompt:
    assert agent_cli.main(argv, stdout=stdout, stderr=stderr) == 1
  assert "identity_unresolved" in stderr.getvalue()
  prompt.assert_not_called()
  factory.assert_not_called()
  assert client.mock_calls == []


@pytest.mark.parametrize("risk_id", [None, 0, True, "bad"])
def test_roster_without_valid_risk_id_is_unresolved(
  door: tuple[MagicMock, MagicMock], monkeypatch: pytest.MonkeyPatch, risk_id: object,
) -> None:
  _, factory = door
  row: dict[str, object] = {"slug": "analyst"}
  if risk_id is not None:
    row["risk_user_id"] = risk_id
  monkeypatch.setenv("GATEWAY_USER_KEYS", json.dumps([row]))
  stderr = StringIO()
  with patch("getpass.getpass") as prompt:
    assert agent_cli.main(
      ["auth", "login", "openai", "--api-key", "--user", "analyst"],
      stdout=StringIO(), stderr=stderr,
    ) == 1
  assert "identity_unresolved" in stderr.getvalue()
  prompt.assert_not_called()
  factory.assert_not_called()


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("code,message", [
  ("key_invalid", "the provider rejected this API key"),
  ("workspace_required", "this key needs a workspace id"),
  ("key_already_enrolled", "this key is already connected"),
])
def test_login_renders_typed_refusals_without_key_material(
  door: tuple[MagicMock, MagicMock], provider: str, code: str, message: str,
  caplog: pytest.LogCaptureFixture,
) -> None:
  client, _ = door
  client.add_key.side_effect = EnrolmentRefused(EnrolmentRefusal.from_wire({
    "contract_version": 2, "error": code, "message": SECRET,
  }))
  stdout, stderr = StringIO(), StringIO()
  with patch("getpass.getpass", return_value=SECRET):
    assert agent_cli.main(
      ["auth", "login", provider, "--api-key", "--user", "analyst"],
      stdout=stdout, stderr=stderr,
    ) == 1
  assert code in stderr.getvalue()
  assert message in stderr.getvalue()
  assert SECRET not in stdout.getvalue() + stderr.getvalue() + caplog.text


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_status_lists_key_health_and_keeps_anthropic_seat_status(
  door: tuple[MagicMock, MagicMock], provider: Provider,
) -> None:
  client, _ = door
  client.list_keys.return_value = EnrolmentOk(731, provider, (key_view(suspended=True),))
  stdout = StringIO()
  assert agent_cli.main(
    ["auth", "status", provider, "--user", "analyst"], stdout=stdout,
  ) == 0
  client.list_keys.assert_called_once_with(risk_user_id=731, provider=provider)
  text = stdout.getvalue()
  for field in ("research", FINGERPRINT, "suspended", VALIDATED, "out_of_credit"):
    assert field in text
  if provider == "anthropic":
    assert "Anthropic OAuth: no enrolled accounts" in text


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("flag,operation", [(None, "remove"), ("--disable", "disable"), ("--enable", "enable")])
def test_logout_mutates_only_named_user_key(
  door: tuple[MagicMock, MagicMock], provider: Provider, flag: str | None, operation: str,
) -> None:
  client, _ = door
  client.set_key_status.return_value = EnrolmentOk(731, provider, ())
  argv = ["auth", "logout", provider, "--user", "analyst", "--key", FINGERPRINT]
  if flag is not None:
    argv.append(flag)
  assert agent_cli.main(argv, stdout=StringIO()) == 0
  client.set_key_status.assert_called_once_with(
    risk_user_id=731, provider=provider, fingerprint=FINGERPRINT, operation=operation,
  )


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("flag", ["--api-key", "--disable", "--enable"])
def test_logout_requires_fingerprint_before_mutation(
  door: tuple[MagicMock, MagicMock], provider: str, flag: str,
) -> None:
  client, _ = door
  stderr = StringIO()
  assert agent_cli.main(
    ["auth", "logout", provider, "--user", "analyst", flag], stdout=StringIO(), stderr=stderr,
  ) == 2
  assert "--key <fingerprint>" in stderr.getvalue()
  client.set_key_status.assert_not_called()


def test_openai_flow_is_registered_user_api_key_kind() -> None:
  assert get_login_flow("openai").kind == "user_api_key"


@pytest.mark.parametrize("user", ["analyst", "ANALYST@example.com"])
def test_anthropic_status_uses_login_roster_binding_for_seat_scope(door, user):
  client, _ = door
  client.list_keys.return_value = EnrolmentOk(731, "anthropic", ())
  with patch("agent_gateway.cli._auth_status_anthropic", return_value=0) as seat_status:
    assert agent_cli.main(
      ["auth", "status", "anthropic", "--user", user], stdout=StringIO(),
    ) == 0
  assert seat_status.call_args.args[0].user == 731
  client.list_keys.assert_called_once_with(risk_user_id=731, provider="anthropic")
