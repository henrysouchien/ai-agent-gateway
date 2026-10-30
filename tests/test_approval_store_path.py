from __future__ import annotations

import asyncio
import hashlib
import hmac
from pathlib import Path

import pytest
import httpx
from hank_contracts.agent_api import AGENT_API_CLAIM_HEADERS, canonical_agent_api_claim_bytes
from fastapi import FastAPI
from pydantic import ValidationError

from agent_gateway.approval_remote import (
  APPROVALS_EGRESS_URL, APPROVALS_EGRESS_URL_ENV, encode_operation_result,
)
from agent_gateway.approval_remote_store import RemoteApprovalRequestStore
from agent_gateway.approval_subsystem import (
  build_runtime_approval_subsystem, load_placed_approval_configuration,
)
from agent_gateway.user_runtime_credentials import (
  USER_RUNTIME_CREDENTIALS_FILE_ENV,
  DeskReplica,
  UserRuntimeIdentity,
  mint_user_runtime_credentials,
  load_user_runtime_credentials,
  write_user_runtime_credentials,
)

from agent_gateway.approval_store import resolve_approval_db_path
from agent_gateway import server_chat_helpers
from agent_gateway.server import GatewayServerConfig


def test_approval_db_path_requires_an_authoritative_state_root() -> None:
  with pytest.raises(
    RuntimeError,
    match="approval database requires USER_DATA_DIR",
  ):
    resolve_approval_db_path(
      env_get=lambda _key, _default: "",
    )


def test_approval_db_path_defaults_beneath_user_data_dir(tmp_path: Path) -> None:
  user_data_dir = tmp_path / "application-data"

  def env_get(key: str, default: str) -> str:
    return str(user_data_dir) if key == "USER_DATA_DIR" else default

  assert resolve_approval_db_path(env_get=env_get) == (
    user_data_dir / "gateway" / "approvals.sqlite3"
  )


def test_approval_db_path_accepts_absolute_campaign_path(tmp_path: Path) -> None:
  user_data_dir = tmp_path / "campaign" / "data"
  expected = user_data_dir / "gateway" / "approvals.sqlite3"

  def env_get(key: str, default: str) -> str:
    values = {
      "GATEWAY_APPROVAL_DB_PATH": str(expected),
      "USER_DATA_DIR": str(user_data_dir),
    }
    return values.get(key, default)

  assert resolve_approval_db_path(
    env_get=env_get,
  ) == expected


def test_approval_db_path_rejects_relative_configuration() -> None:
  def env_get(key: str, default: str) -> str:
    values = {
      "GATEWAY_APPROVAL_DB_PATH": "data/approvals.sqlite3",
      "USER_DATA_DIR": "/var/lib/agent-gateway",
    }
    return values.get(key, default)

  with pytest.raises(ValueError, match="GATEWAY_APPROVAL_DB_PATH must be an absolute path"):
    resolve_approval_db_path(env_get=env_get)


def test_approval_db_path_rejects_relative_user_data_dir() -> None:
  def env_get(key: str, default: str) -> str:
    return "relative-data" if key == "USER_DATA_DIR" else default

  with pytest.raises(ValueError, match="USER_DATA_DIR must be an absolute path"):
    resolve_approval_db_path(env_get=env_get)


def test_explicit_approval_path_overrides_user_data_dir(tmp_path: Path) -> None:
  user_data_dir = tmp_path / "application-data"
  expected = user_data_dir / "gateway" / "approvals.sqlite3"

  def env_get(key: str, default: str) -> str:
    values = {
      "GATEWAY_APPROVAL_DB_PATH": str(expected),
      "USER_DATA_DIR": str(user_data_dir),
    }
    return values.get(key, default)

  assert resolve_approval_db_path(env_get=env_get) == expected


def test_explicit_approval_path_must_match_user_data_root(
  tmp_path: Path,
) -> None:
  def env_get(key: str, default: str) -> str:
    values = {
      "GATEWAY_APPROVAL_DB_PATH": str(
        tmp_path / "other" / "approvals.sqlite3"
      ),
      "USER_DATA_DIR": str(tmp_path / "application-data"),
    }
    return values.get(key, default)

  with pytest.raises(
    ValueError,
    match=(
      "GATEWAY_APPROVAL_DB_PATH must equal "
      "USER_DATA_DIR/gateway/approvals.sqlite3"
    ),
  ):
    resolve_approval_db_path(env_get=env_get)


def test_gateway_approval_subsystem_uses_configured_campaign_db(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  user_data_dir = tmp_path / "campaign" / "data"
  expected = user_data_dir / "gateway" / "approvals.sqlite3"
  # The state root is created by the deployment, not by the subsystem.
  expected.parent.mkdir(parents=True, mode=0o700)
  monkeypatch.setenv("USER_DATA_DIR", str(user_data_dir))
  monkeypatch.setenv("GATEWAY_APPROVAL_DB_PATH", str(expected))
  app = FastAPI()
  config = GatewayServerConfig(
    audit_hmac_secret_resolver=lambda: b"secret",
    audit_hmac_key_id_resolver=lambda: "key-1",
    tool_input_redactor=lambda *_args, **_kwargs: {},
  )

  server_chat_helpers._init_approval_subsystem(app, config)

  assert app.state.gateway_approval_store.path == expected


@pytest.fixture
def placed_approval_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  path = tmp_path / "credentials.json"
  write_user_runtime_credentials(
    mint_user_runtime_credentials(
      org_claim_key="approval-path-test-org-key-at-least-32-bytes",
      identity=UserRuntimeIdentity(
        key="private-api-key", channel="excel", slug="alice",
        email="alice@example.test", risk_user_id=17, role="analyst",
        capabilities=("research",),
      ),
      storage_id="17",
      generation=1,
      desk=DeskReplica("s3://test-desk", None, None, None, None),
    ),
    path,
  )
  monkeypatch.setenv(USER_RUNTIME_CREDENTIALS_FILE_ENV, str(path))
  monkeypatch.setenv("HANK_APPROVAL_TRANSPORT", "local")
  return path


@pytest.mark.parametrize("url", [None, "", "not-a-url", "file:///tmp/approvals"])
def test_placed_approval_entry_requires_http_egress(
  placed_approval_bundle: Path,
  url: str | None,
) -> None:
  environment = {
    USER_RUNTIME_CREDENTIALS_FILE_ENV: str(placed_approval_bundle),
    "HANK_APPROVAL_TRANSPORT": "local",
  }
  if url is not None:
    environment[APPROVALS_EGRESS_URL_ENV] = url
  with pytest.raises(ValidationError):
    load_placed_approval_configuration(environment)


@pytest.mark.parametrize(
  "url", [APPROVALS_EGRESS_URL, "http://127.0.0.1:8000/egress/approvals"],
)
def test_placed_gateway_composes_remote_without_local_ledger(
  placed_approval_bundle: Path,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  url: str,
) -> None:
  state_root = tmp_path / "data"
  monkeypatch.setenv("USER_DATA_DIR", str(state_root))
  monkeypatch.setenv(APPROVALS_EGRESS_URL_ENV, url)
  app = FastAPI()
  server_chat_helpers._init_approval_subsystem(
    app,
    GatewayServerConfig(
      audit_hmac_secret_resolver=lambda: b"secret",
      audit_hmac_key_id_resolver=lambda: "key-1",
    ),
  )
  assert isinstance(app.state.gateway_approval_store, RemoteApprovalRequestStore)
  assert app.state.gateway_approval_remote is True
  assert app.state.gateway_approval_audit_writer is None
  assert not state_root.exists()


@pytest.mark.parametrize("transport", [None, "", "unknown"])
def test_placed_approval_requires_explicit_transport(
  placed_approval_bundle: Path, transport: str | None,
) -> None:
  environment = {
    USER_RUNTIME_CREDENTIALS_FILE_ENV: str(placed_approval_bundle),
    APPROVALS_EGRESS_URL_ENV: APPROVALS_EGRESS_URL,
  }
  if transport is not None:
    environment["HANK_APPROVAL_TRANSPORT"] = transport
  with pytest.raises(ValidationError):
    load_placed_approval_configuration(environment)


@pytest.mark.parametrize("transport", ["local", "agentcore"])
def test_placed_approval_http_authentication_uses_transport_not_port(
  placed_approval_bundle: Path,
  monkeypatch: pytest.MonkeyPatch,
  transport: str,
) -> None:
  # A LocalProcess edge on 8090 has the same URL as the AgentCore tunnel.
  monkeypatch.setenv(APPROVALS_EGRESS_URL_ENV, APPROVALS_EGRESS_URL)
  monkeypatch.setenv("HANK_APPROVAL_TRANSPORT", transport)
  requests: list[httpx.Request] = []

  def respond(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    assert str(request.url) == APPROVALS_EGRESS_URL + "/request.get"
    return httpx.Response(
      200, content=encode_operation_result("request.get", None),
    )

  mock_transport = httpx.MockTransport(respond)
  sync_client = httpx.Client
  async_client = httpx.AsyncClient
  monkeypatch.setattr(
    httpx, "Client",
    lambda **kwargs: sync_client(transport=mock_transport, **kwargs),
  )
  monkeypatch.setattr(
    httpx, "AsyncClient",
    lambda **kwargs: async_client(transport=mock_transport, **kwargs),
  )
  subsystem = build_runtime_approval_subsystem()
  assert asyncio.run(subsystem.store.get("approval-not-found")) is None
  assert len(requests) == 1
  headers = requests[0].headers
  if transport == "agentcore":
    assert not any(name in headers for name in AGENT_API_CLAIM_HEADERS.values())
    return

  claim = {key: headers[name] for key, name in AGENT_API_CLAIM_HEADERS.items()}
  bundle = load_user_runtime_credentials({
    USER_RUNTIME_CREDENTIALS_FILE_ENV: str(placed_approval_bundle),
  })
  assert bundle is not None
  assert claim["user_id"] == bundle.approval_claim_signing.user_id
  assert claim["user_email"] == bundle.approval_claim_signing.user_email
  canonical = canonical_agent_api_claim_bytes(
    audience=claim["audience"], issued_at=int(claim["issued_at"]),
    expiry=int(claim["expiry"]), user_id=claim["user_id"],
    user_email=claim["user_email"], nonce=claim["nonce"],
  )
  approval_signature = hmac.new(
    bundle.approval_claim_signing.key.encode(), canonical, hashlib.sha256,
  ).hexdigest()
  risk_signature = hmac.new(
    bundle.claim_signing.key.encode(), canonical, hashlib.sha256,
  ).hexdigest()
  assert claim["signature"] == approval_signature
  assert claim["signature"] != risk_signature
