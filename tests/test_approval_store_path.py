from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI

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
  # The standing approval preference is approval state: one file beside the
  # ledger, handed to the one policy every channel shares.
  assert app.state.gateway_approval_preference_store.path == (
    user_data_dir / "gateway" / "approval-preferences.sqlite3"
  )
