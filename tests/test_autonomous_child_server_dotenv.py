from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_gateway import autonomous_child_server_dotenv as server_dotenv
from agent_gateway.autonomous_runner_start import _positive_autonomous_child_env
from api.agent.autonomous import mcp_config


_ROOT = Path(__file__).resolve().parents[3]
_RISK_SERVERS = frozenset({
  "fmp-mcp",
  "research-mcp",
  "portfolio-trades-mcp",
  "portfolio-config-mcp",
  "portfolio-producers-mcp",
  "portfolio-writes-mcp",
  "portfolio-reads-mcp",
})


@pytest.mark.parametrize("template_name", [
  "mcp.production-cron.template.json",
  "mcp.local-cron.template.json",
])
@pytest.mark.parametrize(("gateway_verify", "sibling_verify"), [
  (None, "true"),
  (None, "false"),
  ("true", "false"),
  ("false", "true"),
])
def test_cron_templates_render_declared_tls_setting_without_missing_env(
  template_name: str,
  gateway_verify: str | None,
  sibling_verify: str,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  template = json.loads((_ROOT / "deploy" / template_name).read_text(encoding="utf-8"))
  # Keep the deployed server/env declarations; rebase only their application
  # cwd so the parent reads fixture dotfiles, never host credentials.
  for config in template["mcpServers"].values():
    config["cwd"] = str(tmp_path / Path(config["cwd"]).name)
  template_path = tmp_path / "template.json"
  template_path.write_text(json.dumps(template), encoding="utf-8")
  risk_root = tmp_path / "risk_module"
  risk_root.mkdir()
  authority_values = {
    "GATEWAY_USER_KEYS": "sibling-user-key-table",
    "AGENT_API_USER_CLAIM_HMAC_KEY": "sibling-claim-signing-key",
    "GATEWAY_SUBJECT_ASSERTION_ED25519_PRIVATE_KEY": "sibling-assertion-signing-key",
    "ANTHROPIC_API_KEY": "sibling-provider-key",
    "FUTURE_GATEWAY_SECRET": "sibling-future-secret",
  }
  (risk_root / ".env").write_text(
    f"GATEWAY_SSL_VERIFY={sibling_verify}\n"
    + "".join(f"{name}={value}\n" for name, value in authority_values.items()),
    encoding="utf-8",
  )
  monkeypatch.setattr(server_dotenv, "_OWN_APPLICATION_ROOT", tmp_path / "ai-excel-addin")
  parent_env = {
    "MCP_CONFIG_TEMPLATE": str(template_path),
    "MCP_CONFIG_TMP_DIR": str(tmp_path / "mcp"),
    "JUPYTER_ROOT": str(tmp_path),
    "USER_DATA_DIR": str(tmp_path / "data"),
    "ENVIRONMENT": "production",
    "GATEWAY_BASE_URL": "https://gateway.example.test",
  }
  if gateway_verify is not None:
    parent_env["GATEWAY_SSL_VERIFY"] = gateway_verify
  child_env = _positive_autonomous_child_env(
    parent_env,
    provider="anthropic",
    profile="analyst",
    deliver=False,
  )
  # The parent-selected identity normally supplies these at materialization;
  # no authority value below may be recovered from a sibling's dotenv.
  monkeypatch.setattr(os, "environ", child_env)
  monkeypatch.setattr(mcp_config.user_identity, "resolve_storage_user_id", lambda *_: "1")
  monkeypatch.setattr(mcp_config, "_resolve_gateway_api_key", lambda *_: "selected-mcp-key")
  monkeypatch.setattr(mcp_config, "_PRIVATE_TMP", tmp_path / "rendered")

  render = mcp_config._materialize_mcp_config("1", "alice@example.com", "analyst")
  servers = json.loads(render.path.read_text(encoding="utf-8"))["mcpServers"]
  expected_verify = gateway_verify if gateway_verify is not None else sibling_verify
  assert {
    name: config["env"]["GATEWAY_SSL_VERIFY"]
    for name, config in servers.items()
    if name in _RISK_SERVERS
  } == dict.fromkeys(_RISK_SERVERS, expected_verify)
  assert all("GATEWAY_SSL_VERIFY" not in names for names in render.unresolved_env.values())
  delivered = server_dotenv.decode_server_dotenv(
    child_env.get(server_dotenv.AUTONOMOUS_CHILD_SERVER_DOTENV_ENV)
  )
  for config in servers.values():
    if "GATEWAY_API_KEY" in config["env"]:
      assert config["env"]["GATEWAY_API_KEY"] == "selected-mcp-key"
    assert not authority_values.keys() & config["env"].keys()
  for values in delivered.values():
    assert not authority_values.keys() & values.keys()
  captured = capsys.readouterr()
  assert all(
    "GATEWAY_SSL_VERIFY" not in line
    for line in captured.err.splitlines()
    if "missing env:" in line
  )
  for secret in authority_values.values():
    assert secret not in "\n".join(child_env.values())
    assert secret not in captured.out + captured.err


@pytest.mark.parametrize("gateway_origin", [
  "https://localhost:8534",
  "https://localhost:8000",
])
def test_local_topology_keeps_authenticated_child_on_its_selected_gateway(
  gateway_origin: str,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  template = json.loads((_ROOT / "deploy" / "mcp.local.template.json").read_text(encoding="utf-8"))
  gateway_consumers = {
    name for name, config in template["mcpServers"].items()
    if "GATEWAY_URL" in config.get("env", {})
  }
  assert "portfolio-producers-mcp" in gateway_consumers
  risk_root = tmp_path / "risk_module"
  risk_root.mkdir()
  sibling_origin = "https://wrong-gateway.example.test"
  (risk_root / ".env").write_text(f"GATEWAY_URL={sibling_origin}\n", encoding="utf-8")
  parent_env = {
    "MCP_CONFIG_TEMPLATE": str(_ROOT / "deploy" / "mcp.local.template.json"),
    "MCP_CONFIG_TMP_DIR": str(tmp_path / "mcp"),
    "JUPYTER_ROOT": str(tmp_path),
    "EDGAR_UPDATER_ROOT": str(tmp_path / "edgar_updater"),
    "RISK_MODULE_DOTENV": str(risk_root / ".env"),
    "LOCAL_GATEWAY_RUNTIME_ROOT": str(tmp_path / "runtime"),
    "LOCAL_GATEWAY_RUNTIME_VERSION_ROOT": str(tmp_path / "runtime" / "version"),
    "USER_DATA_DIR": str(tmp_path / "data"),
    "GATEWAY_BASE_URL": gateway_origin,
    "GATEWAY_URL": sibling_origin,
  }
  monkeypatch.setattr(server_dotenv, "_OWN_APPLICATION_ROOT", tmp_path / "ai-excel-addin")
  child_env = _positive_autonomous_child_env(
    parent_env, provider="anthropic", profile="analyst", deliver=False,
  )
  assert child_env["GATEWAY_BASE_URL"] == gateway_origin
  assert "GATEWAY_URL" not in child_env
  monkeypatch.setattr(os, "environ", child_env)
  monkeypatch.setattr(mcp_config.user_identity, "resolve_storage_user_id", lambda *_: "1")
  monkeypatch.setattr(mcp_config, "_resolve_gateway_api_key", lambda *_: "selected-mcp-key")
  monkeypatch.setattr(mcp_config, "_PRIVATE_TMP", tmp_path / "rendered")
  monkeypatch.setattr(mcp_config, "_local_provider_dotenv_paths", lambda *_: (risk_root / ".env",))
  render = mcp_config._materialize_mcp_config("1", "alice@example.com", "analyst")
  servers = json.loads(render.path.read_text(encoding="utf-8"))["mcpServers"]
  assert {
    name: servers[name]["env"]["GATEWAY_URL"] for name in gateway_consumers
  } == dict.fromkeys(gateway_consumers, gateway_origin)
