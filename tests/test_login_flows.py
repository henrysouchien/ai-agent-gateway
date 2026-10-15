from __future__ import annotations

import argparse
from io import StringIO
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch
import hashlib
import base64
import urllib.parse

import httpx
import pytest

from agent_gateway import cli as agent_cli
AgentProjectError = agent_cli.AgentProjectError
from agent_gateway.login_flows import (
  GeminiLoginFlow,
  GoogleLoginFlow,
  CopilotLoginFlow,
  MuseLoginFlow,
  OllamaLoginFlow,
  OllamaCloudLoginFlow,
  registered_login_flows,
  get_login_flow,
  run_loopback_oauth_flow,
)
from agent_gateway.providers import (
  google_oauth,
  copilot_oauth,
  google_antigravity_oauth,
  muse_oauth,
  api_key_store,
  kimi_code_oauth,
)


def test_auth_login_and_status_choices_equal_registered_families() -> None:
  parser = agent_cli.build_parser()
  subparsers_action = next(
    action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
  )
  auth_parser = subparsers_action.choices["auth"]
  auth_sub_action = next(
    action for action in auth_parser._actions if isinstance(action, argparse._SubParsersAction)
  )

  registered_families = set(registered_login_flows().keys())
  expected_subset = {"anthropic", "codex", "copilot", "gemini", "google", "kimi-code", "muse", "ollama", "ollama-cloud", "openai", "xai"}
  assert expected_subset.issubset(registered_families)

  login_parser = auth_sub_action.choices["login"]
  login_provider_action = next(a for a in login_parser._actions if a.dest == "provider")
  assert set(login_provider_action.choices) == registered_families

  status_parser = auth_sub_action.choices["status"]
  status_provider_action = next(a for a in status_parser._actions if a.dest == "provider")
  assert set(status_provider_action.choices) == registered_families

  logout_parser = auth_sub_action.choices["logout"]
  logout_provider_action = next(a for a in logout_parser._actions if a.dest == "provider")
  assert set(logout_provider_action.choices) == registered_families


def test_gemini_subscription_oauth_writes_store_and_discovers_project(tmp_path: Path) -> None:
  store = tmp_path / "antigravity.json"

  def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/token":
      data = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
      assert data.get("grant_type") == "authorization_code"
      assert data.get("code") == "test-gemini-code"
      return httpx.Response(
        200,
        json={
          "access_token": "ya29.antigravity-token",
          "expires_in": 3600,
          "refresh_token": "1//antigravity-refresh",
          "token_type": "Bearer",
        },
      )
    if request.url.path == "/v1internal:loadCodeAssist":
      return httpx.Response(
        200,
        json={
          "currentTier": {"id": "free-tier"},
          "cloudaicompanionProject": "projects/test-antigravity-proj-123",
        },
      )
    return httpx.Response(404, text="Not Found")

  mock_transport = httpx.MockTransport(handler)

  def fake_loopback_flow(*, auth_url_builder, token_exchange, stdout=None, no_browser=False, port=0):
    real_client = httpx.Client
    with patch("httpx.Client", side_effect=lambda **kwargs: real_client(transport=mock_transport, **kwargs)):
      return token_exchange("test-gemini-code", "test-verifier", "http://127.0.0.1:51121/oauth-callback")

  with patch("agent_gateway.login_flows.run_loopback_oauth_flow", side_effect=fake_loopback_flow):
    stdout = StringIO()
    ret = agent_cli.main(["auth", "login", "gemini", "--no-browser", "--store", str(store)], stdout=stdout)
    assert ret == 0
    assert "Gemini Antigravity OAuth login complete" in stdout.getvalue()

  record = google_antigravity_oauth.load_antigravity_token_record(store)
  assert record is not None
  assert record["access_token"] == "ya29.antigravity-token"
  assert record["project_id"] == "projects/test-antigravity-proj-123"
  assert record["source"] == "subscription"


def test_gemini_api_key_prompt_writes_store(tmp_path: Path) -> None:
  store = tmp_path / "google_api_key.json"
  with patch("getpass.getpass", return_value="AIzaSyFakeGeminiKey123"):
    stdout = StringIO()
    ret = agent_cli.main(["auth", "login", "gemini", "--api-key", "--no-browser", "--store", str(store)], stdout=stdout)
    assert ret == 0
    assert "Gemini (Google AI Studio) API key saved" in stdout.getvalue()

  record = api_key_store.load_api_key_record(store)
  assert record is not None
  assert record["api_key"] == "AIzaSyFakeGeminiKey123"
  assert record["source"] == "api_key"


def test_muse_subscription_device_code_flow_writes_store(tmp_path: Path) -> None:
  store = tmp_path / "muse_credentials.json"

  def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/oidc/device/authorization/":
      return httpx.Response(
        200,
        json={
          "device_code": "dc-muse-123",
          "user_code": "MUSE-4567",
          "verification_uri": "https://auth.meta.com/oidc/device/authorization/",
          "verification_uri_complete": "https://auth.meta.com/oidc/device/authorization/?user_code=MUSE-4567",
          "expires_in": 900,
          "interval": 0.01,
        },
      )
    if request.url.path == "/oidc/device/token/":
      return httpx.Response(
        200,
        json={
          "access_token": "meta-access-token-abc",
          "token_type": "bearer",
        },
      )
    if request.url.path == "/muse-code/key":
      assert request.headers.get("Authorization") == "Bearer meta-access-token-abc"
      assert request.headers.get("x-api-version") == "1.0.0"
      return httpx.Response(
        200,
        json={
          "api_key": "meta_model_api_key_secret_999",
          "user_id": "usr_meta_test",
          "user_email": "engineer@meta.com",
          "is_subs_active": True,
        },
      )
    return httpx.Response(404, text="Not Found")

  mock_transport = httpx.MockTransport(handler)
  real_client = httpx.Client
  with patch("httpx.Client", side_effect=lambda **kwargs: real_client(transport=mock_transport, **kwargs)):
    stdout = StringIO()
    ret = agent_cli.main(["auth", "login", "muse", "--no-browser", "--store", str(store)], stdout=stdout)
    assert ret == 0
    assert "Muse device authorization complete" in stdout.getvalue()

  record = muse_oauth.load_muse_token_record(store)
  assert record is not None
  assert record["api_key"] == "meta_model_api_key_secret_999"
  assert record["source"] == "subscription"
  assert record["email"] == "engineer@meta.com"


def test_muse_api_key_prompt_writes_store(tmp_path: Path) -> None:
  store = tmp_path / "muse_api_key.json"
  with patch("getpass.getpass", return_value="direct_meta_key_777"):
    stdout = StringIO()
    ret = agent_cli.main(["auth", "login", "muse", "--api-key", "--no-browser", "--store", str(store)], stdout=stdout)
    assert ret == 0
    assert "Muse (Meta Model API) API key saved" in stdout.getvalue()

  record = api_key_store.load_api_key_record(store)
  assert record is not None
  assert record["api_key"] == "direct_meta_key_777"
  assert record["source"] == "api_key"


def test_muse_api_key_status_and_logout_target_the_api_key_store(tmp_path: Path) -> None:
  subscription_store = tmp_path / "muse" / "credentials.json"
  api_store = tmp_path / "muse_api_key.json"
  muse_oauth.save_muse_token_record(
    subscription_store,
    {"api_key": "subscription_key_111", "source": "subscription", "email": "sub@meta.com"},
  )
  api_key_store.save_api_key_record(
    api_store, {"api_key": "api_seat_key_222", "source": "api_key", "identity": "api-seat"}
  )
  with patch.dict("os.environ", {"MUSE_AUTH_STORE_PATH": str(subscription_store)}, clear=False):
    stdout = StringIO()
    ret = agent_cli.main(
      ["auth", "status", "muse", "--api-key", "--store", str(api_store)], stdout=stdout
    )
    assert ret == 0
    assert "api_seat_key_222" not in stdout.getvalue()
    assert "api-seat" in stdout.getvalue()
    stdout = StringIO()
    ret = agent_cli.main(
      ["auth", "logout", "muse", "--api-key", "--store", str(api_store), "--account", "api-seat"],
      stdout=stdout,
    )
    assert ret == 0
  assert api_key_store.load_api_key_record(api_store) is None
  subscription_record = muse_oauth.load_muse_token_record(subscription_store)
  assert subscription_record is not None
  assert subscription_record["api_key"] == "subscription_key_111"


def _kimi_jwt(email: str, nonce: str) -> str:
  def segment(value: dict[str, str]) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode("utf-8")).decode("ascii").rstrip("=")

  return f"{segment({'alg': 'none'})}.{segment({'email': email, 'nonce': nonce})}.signature"


def test_kimi_code_sequential_logins_preserve_seats_and_status_reports_both(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  for name in ("KIMI_CODE_AUTH_TOKEN", "KIMI_CODE_API_KEY", "KIMI_API_KEY"):
    monkeypatch.delenv(name, raising=False)
  store = tmp_path / "kimi_code" / "oauth.json"
  seats = [
    {
      "access_token": _kimi_jwt(f"seat-{n}@example.com", f"access-{n}"),
      "refresh_token": f"kimi-refresh-secret-{n}",
      "expires_at": time.time() + 3600,
      "source": "subscription",
    }
    for n in (1, 2)
  ]
  devices = [
    {
      "user_code": f"KIMI-000{n}",
      "device_code": f"device-code-{n}",
      "verification_uri": "https://www.kimi.com/code/authorize_device",
      "interval": 1,
      "expires_in": 600,
    }
    for n in (1, 2)
  ]

  with patch(
    "agent_gateway.login_flows.request_kimi_code_device_code", side_effect=devices,
  ) as request_device, patch(
    "agent_gateway.login_flows.poll_kimi_code_device_token", side_effect=[dict(s) for s in seats],
  ) as poll_device:
    for n in (1, 2):
      stdout = StringIO()
      ret = agent_cli.main(
        ["auth", "login", "kimi-code", "--no-browser", "--store", str(store)], stdout=stdout,
      )
      assert ret == 0
      assert "Kimi Code device authorization complete" in stdout.getvalue()
      assert f"accounts={n}" in stdout.getvalue()

  assert request_device.call_count == 2
  assert [call.args[0] for call in poll_device.call_args_list] == ["device-code-1", "device-code-2"]

  accounts = json.loads(store.read_text(encoding="utf-8"))["accounts"]
  assert [(a["identity"], a["access_token"], a["refresh_token"]) for a in accounts] == [
    ("seat-1@example.com", seats[0]["access_token"], "kimi-refresh-secret-1"),
    ("seat-2@example.com", seats[1]["access_token"], "kimi-refresh-secret-2"),
  ]
  assert len(kimi_code_oauth.load_kimi_code_accounts(store)) == 2

  stdout = StringIO()
  assert agent_cli.main(["auth", "status", "kimi-code", "--store", str(store)], stdout=stdout) == 0
  status = stdout.getvalue()
  assert "Kimi Code (subscription): active" in status
  assert "accounts=2" in status
  for n in (1, 2):
    assert f"seat-{n}@example.com" in status
    assert seats[n - 1]["access_token"] not in status
    assert f"kimi-refresh-secret-{n}" not in status


def test_kimi_code_api_key_prompt_writes_credentials_store(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  for name in ("KIMI_CODE_API_KEY_STORE_PATH", "KIMI_CODE_AUTH_STORE_PATH", "KIMI_CODE_AUTH_TOKEN", "KIMI_CODE_API_KEY", "KIMI_API_KEY"):
    monkeypatch.delenv(name, raising=False)
  store = tmp_path / "kimi_code" / "credentials.json"
  with patch("getpass.getpass", return_value="kimi_secret_api_key_555"):
    stdout = StringIO()
    ret = agent_cli.main(["auth", "login", "kimi-code", "--api-key", "--no-browser"], stdout=stdout)
    assert ret == 0
    assert "Kimi Code API key saved" in stdout.getvalue()

  record = api_key_store.load_api_key_record(store)
  assert record is not None
  assert record["api_key"] == "kimi_secret_api_key_555"
  assert record["source"] == "api_key"
  assert not (tmp_path / "kimi_code" / "oauth.json").exists()

  stdout = StringIO()
  assert agent_cli.main(["auth", "status", "kimi-code", "--api-key"], stdout=stdout) == 0
  status = stdout.getvalue()
  assert "Kimi Code (API key): active" in status
  assert "accounts=1" in status
  assert "kimi_secret_api_key_555" not in status


def test_ollama_cloud_api_key_prompt_writes_store(tmp_path: Path) -> None:
  store = tmp_path / "ollama_cloud_key.json"
  with patch("getpass.getpass", return_value="ollama_secret_cloud_key_444"):
    stdout = StringIO()
    ret = agent_cli.main(["auth", "login", "ollama-cloud", "--no-browser", "--store", str(store)], stdout=stdout)
    assert ret == 0
    assert "Ollama Cloud API key saved" in stdout.getvalue()

  record = api_key_store.load_api_key_record(store)
  assert record is not None
  assert record["api_key"] == "ollama_secret_cloud_key_444"


def test_copilot_device_code_flow_writes_store(tmp_path: Path) -> None:
  store = tmp_path / "copilot_oauth.json"

  def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/login/device/code":
      return httpx.Response(
        200,
        json={
          "device_code": "dc-copilot-123",
          "user_code": "ABCD-1234",
          "verification_uri": "https://github.com/login/device",
          "expires_in": 900,
          "interval": 0.01,
        },
      )
    if request.url.path == "/login/oauth/access_token":
      data = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
      assert data.get("device_code") == "dc-copilot-123"
      return httpx.Response(
        200,
        json={
          "access_token": "ghu_copilot_token_xyz",
          "token_type": "bearer",
          "scope": "read:user",
        },
      )
    return httpx.Response(404, text="Not Found")

  mock_transport = httpx.MockTransport(handler)

  with patch.dict(
    "os.environ",
    {
      "COPILOT_OAUTH_CLIENT_ID": "fake-copilot-client-id",
      "COPILOT_OAUTH_DEVICE_CODE_URL": "http://testserver/login/device/code",
      "COPILOT_OAUTH_TOKEN_URL": "http://testserver/login/oauth/access_token",
    },
  ):
    real_client = httpx.Client
    with patch("httpx.Client", side_effect=lambda **kwargs: real_client(transport=mock_transport, **kwargs)):
      stdout = StringIO()
      assert agent_cli.main(["auth", "login", "copilot", "--no-browser", "--store", str(store)], stdout=stdout) == 0
      assert "Copilot device authorization complete" in stdout.getvalue()

  record = copilot_oauth.load_copilot_token_record(store)
  assert record is not None
  assert record["access_token"] == "ghu_copilot_token_xyz"


def test_ollama_delegated_flow_invokes_ollama_signin() -> None:
  with patch("subprocess.run") as mock_run:
    mock_run.return_value = MagicMock(returncode=0)
    stdout = StringIO()
    assert agent_cli.main(["auth", "login", "ollama"], stdout=stdout) == 0
    mock_run.assert_called_once_with(["ollama", "signin"], check=False)
    assert "Delegated signin with `ollama` complete." in stdout.getvalue()

  with patch("subprocess.run") as mock_run:
    mock_run.return_value = MagicMock(returncode=1)
    stdout = StringIO()
    stderr = StringIO()
    assert agent_cli.main(["auth", "login", "ollama"], stdout=stdout, stderr=stderr) == 1
    assert "Delegated signin with `ollama` exited with code 1." in stdout.getvalue()


def test_auth_status_prints_identity_and_no_secret(tmp_path: Path) -> None:
  # Antigravity
  ag_store = tmp_path / "antigravity.json"
  google_antigravity_oauth.save_antigravity_token_record(
    ag_store,
    {"access_token": "ya29.secret_token", "project_id": "proj-999", "source": "subscription"},
  )
  stdout = StringIO()
  assert agent_cli.main(["auth", "status", "gemini", "--store", str(ag_store)], stdout=stdout) == 0
  out = stdout.getvalue()
  assert "Gemini (Antigravity subscription): active" in out
  assert "project=proj-999" in out
  assert "ya29.secret_token" not in out

  # Muse
  muse_store = tmp_path / "muse.json"
  muse_oauth.save_muse_token_record(
    muse_store,
    {"api_key": "meta_secret_key_123", "source": "subscription", "email": "test@meta.com"},
  )
  stdout = StringIO()
  assert agent_cli.main(["auth", "status", "muse", "--store", str(muse_store)], stdout=stdout) == 0
  out = stdout.getvalue()
  assert "Muse (subscription): active" in out
  assert "test@meta.com" in out
  assert "meta_secret_key_123" not in out

  # Ollama Cloud
  oc_store = tmp_path / "ollama_cloud.json"
  api_key_store.save_api_key_record(
    oc_store,
    {"api_key": "ollama_cloud_secret_key", "source": "api_key"},
  )
  stdout = StringIO()
  assert agent_cli.main(["auth", "status", "ollama-cloud", "--store", str(oc_store)], stdout=stdout) == 0
  out = stdout.getvalue()
  assert "Ollama Cloud: active" in out
  assert "ollama_cloud_secret_key" not in out


def test_legacy_flows_dispatch_to_existing_functions() -> None:
  # OpenAI
  stdout = StringIO()
  stderr = StringIO()
  assert agent_cli.main(["auth", "login", "openai"], stdout=stdout, stderr=stderr) == 2
  assert "The OpenAI API does not provide ChatGPT subscription OAuth login" in stderr.getvalue()

  # Codex
  with patch("agent_gateway.cli._auth_login_codex", return_value=0) as mock_codex:
    stdout = StringIO()
    assert agent_cli.main(["auth", "login", "codex"], stdout=stdout) == 0
    mock_codex.assert_called_once()

  # Anthropic
  with patch("agent_gateway.cli._auth_login_anthropic", return_value=0) as mock_anthropic:
    stdout = StringIO()
    assert agent_cli.main(["auth", "login", "anthropic"], stdout=stdout) == 0
    mock_anthropic.assert_called_once()


@pytest.mark.parametrize("callback_state", [None, "wrong-state"])
def test_loopback_oauth_rejects_uncorrelated_callback(callback_state: str | None) -> None:
  exchange = MagicMock()

  def build_auth_url(redirect_uri: str, state: str, challenge: str) -> str:
    params = {"code": "fixture-code"}
    if callback_state is not None:
      params["state"] = callback_state
    with httpx.Client(trust_env=False) as client:
      response = client.get(f"{redirect_uri}?{urllib.parse.urlencode(params)}", timeout=5)
    assert response.status_code == 200
    return "https://accounts.example.com/authorize"

  with pytest.raises(RuntimeError, match="OAuth state mismatch in callback; authorization rejected"):
    run_loopback_oauth_flow(
      auth_url_builder=build_auth_url, token_exchange=exchange,
      no_browser=True, console_fallback=False, timeout=5,
    )
  exchange.assert_not_called()


def test_gemini_sequential_logins_preserve_accounts_and_status_reports_both(tmp_path: Path) -> None:
  store = tmp_path / "google_antigravity" / "oauth.json"
  args = argparse.Namespace(store=str(store), api_key=False, no_browser=True, user=None, email=None)
  enrollment = 0

  def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/token":
      data = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
      assert data["grant_type"] == "authorization_code"
      assert data["client_secret"] == google_antigravity_oauth.ANTIGRAVITY_CLIENT_SECRET
      assert "code_verifier" not in data
      return httpx.Response(200, json={
        "access_token": f"token-{enrollment}", "refresh_token": f"refresh-{enrollment}",
        "expires_in": 3600, "token_type": "Bearer",
      })
    if request.url.path == "/v1internal:loadCodeAssist":
      assert request.headers["Authorization"] == f"Bearer token-{enrollment}"
      return httpx.Response(200, json={
        "currentTier": {"id": "free-tier"}, "cloudaicompanionProject": f"project-{enrollment}",
      })
    if request.url.path == "/oauth2/v2/userinfo":
      assert request.headers["Authorization"] == f"Bearer token-{enrollment}"
      return httpx.Response(200, json={"email": f"seat-{enrollment}@example.com"})
    return httpx.Response(404)

  transport = httpx.MockTransport(handler)
  real_client = httpx.Client

  def loopback(*, auth_url_builder, token_exchange, **kwargs):
    auth_query = urllib.parse.parse_qs(urllib.parse.urlsplit(
      auth_url_builder("http://127.0.0.1:51121/oauth-callback", "state", "challenge"),
    ).query)
    assert "code_challenge" not in auth_query
    with patch("httpx.Client", side_effect=lambda **options: real_client(transport=transport, **options)):
      return token_exchange("code", "unused-verifier", "http://127.0.0.1:51121/oauth-callback")

  with patch("agent_gateway.login_flows.run_loopback_oauth_flow", side_effect=loopback):
    for enrollment in (1, 2):
      assert GeminiLoginFlow().login(args, stdout=StringIO()) == 0

  accounts = json.loads(store.read_text(encoding="utf-8"))["accounts"]
  assert len(accounts) == 2
  assert [(seat["identity"], seat["project_id"], seat["access_token"]) for seat in accounts] == [
    ("seat-1@example.com", "project-1", "token-1"),
    ("seat-2@example.com", "project-2", "token-2"),
  ]
  stdout = StringIO()
  assert GeminiLoginFlow().status(args, stdout=stdout) == 0
  status = stdout.getvalue()
  assert "accounts=2" in status
  for enrollment in (1, 2):
    assert f"seat-{enrollment}@example.com" in status
    assert f"project-{enrollment}" in status
    assert f"token-{enrollment}" not in status
    assert f"refresh-{enrollment}" not in status


def test_public_google_pkce_login_and_refresh_rotate_stored_tokens(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  store = tmp_path / "google" / "oauth.json"
  monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "fixture-google-client")
  monkeypatch.setenv("GOOGLE_OAUTH_TOKEN_URL", "https://google.example/token")
  verifier = "fixture-verifier-with-enough-randomness"
  challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")
  grants: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    data = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
    grants.append(data["grant_type"])
    if data["grant_type"] == "authorization_code":
      assert data["code_verifier"] == verifier
      assert data["code"] == "fixture-google-code"
      return httpx.Response(200, json={
        "access_token": "google-enrolled", "refresh_token": "google-refresh-one", "expires_in": 3600,
      })
    assert data["grant_type"] == "refresh_token"
    assert data["refresh_token"] == "google-refresh-one"
    return httpx.Response(200, json={
      "access_token": "google-refreshed", "refresh_token": "google-refresh-two", "expires_in": 7200,
    })

  transport = httpx.MockTransport(handler)
  real_client = httpx.Client

  def loopback(*, auth_url_builder, token_exchange, **kwargs):
    redirect_uri = "http://127.0.0.1:54321/oauth-callback"
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(auth_url_builder(redirect_uri, "state", challenge)).query)
    assert query["code_challenge"] == [challenge]
    assert query["code_challenge_method"] == ["S256"]
    assert query["state"] == ["state"]
    with patch("httpx.Client", side_effect=lambda **options: real_client(transport=transport, **options)):
      return token_exchange("fixture-google-code", verifier, redirect_uri)

  args = argparse.Namespace(store=str(store), api_key=False, no_browser=True, user=None)
  assert GoogleLoginFlow.kind == "oauth_code"
  with patch("agent_gateway.login_flows.run_loopback_oauth_flow", side_effect=loopback):
    assert GoogleLoginFlow().login(args, stdout=StringIO()) == 0
  enrolled = google_oauth.load_google_token_record(store)
  assert enrolled is not None
  assert enrolled["access_token"] == "google-enrolled"
  settings = google_oauth.resolve_google_oauth_settings({"auth_store_path": str(store)})
  with real_client(transport=transport) as client:
    refreshed = google_oauth.refresh_google_tokens(settings, enrolled, client=client)
  assert refreshed["access_token"] == "google-refreshed"
  saved = google_oauth.load_google_token_record(store)
  assert saved is not None
  assert saved["access_token"] == "google-refreshed"
  assert saved["refresh_token"] == "google-refresh-two"
  assert saved["expires_at"] > enrolled["expires_at"]
  assert len(json.loads(store.read_text(encoding="utf-8"))["accounts"]) == 1
  assert grants == ["authorization_code", "refresh_token"]


def test_google_oauth_store_override_does_not_override_api_key_store(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  oauth_store = tmp_path / "custom-oauth.json"
  api_store = tmp_path / "custom-api-key.json"
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  monkeypatch.setenv("GOOGLE_AUTH_STORE_PATH", str(oauth_store))
  monkeypatch.delenv("GOOGLE_API_KEY_STORE_PATH", raising=False)
  monkeypatch.delenv("GEMINI_API_KEY_STORE_PATH", raising=False)
  assert google_oauth.resolve_google_oauth_settings().store_path == oauth_store
  assert api_key_store.resolve_api_key_store_path("google") == tmp_path / "google" / "credentials.json"
  monkeypatch.setenv("GOOGLE_API_KEY_STORE_PATH", str(api_store))
  assert api_key_store.resolve_api_key_store_path("google") == api_store
  google_oauth.save_google_token_record(oauth_store, {"access_token": "oauth-token"})
  api_key_store.save_api_key_record(api_store, {"api_key": "api-key"})
  oauth_record = google_oauth.load_google_token_record(oauth_store)
  api_record = api_key_store.load_api_key_record(api_store)
  assert oauth_record is not None and oauth_record["access_token"] == "oauth-token"
  assert api_record is not None and api_record["api_key"] == "api-key"
