from pathlib import Path

import pytest

from model_authority.loader import PACKAGED_AUTHORITY_DIR, load_configured_model_authority
from model_authority.run_route import read_entry_run_route, serialize_run_route
from model_authority.selection import project_fallback_chain

from agent_gateway.child_environment import (
  AUTONOMOUS_CHILD_ENVIRONMENT,
  AUTONOMOUS_CHILD_PROFILES,
  CODE_EXECUTE_ENVIRONMENT,
  RUN_BASH_ENVIRONMENT,
  filter_child_environment,
)
from agent_gateway.autonomous_runner_start import _positive_autonomous_child_env
from agent_gateway.mcp_client_config import build_mcp_env


@pytest.mark.parametrize(
  "purpose",
  [AUTONOMOUS_CHILD_ENVIRONMENT, RUN_BASH_ENVIRONMENT, CODE_EXECUTE_ENVIRONMENT],
)
def test_model_routing_is_only_projected_to_autonomous_children(
  purpose: str, tmp_path: Path,
) -> None:
  endpoints = {
    "OLLAMA_CLOUD_BASE_URL": "http://127.0.0.1:18081",
    "OLLAMA_BASE_URL": "http://127.0.0.1:18082",
  }
  projected = filter_child_environment(
    {
      **endpoints,
      "HANK_RUN_ROUTE_FILE": str(_route_file(tmp_path)),
      "OLLAMA_CLOUD_API_KEY": "production-key-must-not-travel",
    },
    purpose=purpose,
  )
  assert projected == (endpoints if purpose == AUTONOMOUS_CHILD_ENVIRONMENT else {})
  assert "HANK_RUN_ROUTE_FILE" not in projected


def test_autonomous_child_profile_limits_carry_no_turn_or_cost_ceiling() -> None:
  env: dict[str, str] = {}
  for profile in AUTONOMOUS_CHILD_PROFILES:
    prefix = f"{profile.upper()}_AGENT_"
    env[f"{prefix}MAX_TOKENS"] = "24000"
    env[f"{prefix}MAX_TURNS"] = "40"
    env[f"{prefix}MAX_BUDGET_USD"] = "6.0"

  projected = filter_child_environment(env, purpose=AUTONOMOUS_CHILD_ENVIRONMENT)

  assert projected == {
    f"{profile.upper()}_AGENT_MAX_TOKENS": "24000"
    for profile in AUTONOMOUS_CHILD_PROFILES
  }


@pytest.mark.parametrize("setting", ["0", "1", None])
def test_child_loads_the_parents_fallback_selection(setting: str | None) -> None:
  env = {
    "HANK_MODEL_AUTHORITY_DIR": str(PACKAGED_AUTHORITY_DIR),
    "ANTHROPIC_API_KEY": "must-not-be-inherited",
    "HANK_UNRELATED_CONTROL": "must-not-be-inherited",
  }
  if setting is not None:
    env["HANK_LOCAL_FALLBACK_ENABLED"] = setting
  parent = load_configured_model_authority(env)
  projected = filter_child_environment(env, purpose=AUTONOMOUS_CHILD_ENVIRONMENT)
  child = load_configured_model_authority(projected)
  enabled = load_configured_model_authority({
    **env, "HANK_LOCAL_FALLBACK_ENABLED": "1",
  })
  disabled = load_configured_model_authority({
    **env, "HANK_LOCAL_FALLBACK_ENABLED": "0",
  })
  changed = False
  for capability in parent.selection_policy.capabilities:
    for role in parent.selection_policy.tiers:
      parent_chain = project_fallback_chain(
        parent.registry, parent.selection_policy, capability, role,
      )
      child_chain = project_fallback_chain(
        child.registry, child.selection_policy, capability, role,
      )
      enabled_chain = project_fallback_chain(
        enabled.registry, enabled.selection_policy, capability, role,
      )
      disabled_chain = project_fallback_chain(
        disabled.registry, disabled.selection_policy, capability, role,
      )
      assert child_chain == parent_chain
      changed |= enabled_chain.entries != disabled_chain.entries
  assert changed, "the loaded authority must exercise local fallback selection"
  assert "ANTHROPIC_API_KEY" not in projected
  assert "HANK_UNRELATED_CONTROL" not in projected


def test_child_inherits_the_parents_handed_run_route_preset(tmp_path: Path) -> None:
  env = {
    "HANK_MODEL_AUTHORITY_DIR": str(PACKAGED_AUTHORITY_DIR),
    "HANK_RUN_ROUTE_FILE": str(_route_file(tmp_path)),
  }
  parent = load_configured_model_authority(env)
  projected = _autonomous_child_env(env, env_projection=None)
  child = load_configured_model_authority(projected)

  assert parent.selection_policy.default_preset == "kimi_dev"
  assert parent.selection_policy.default_preset_source == "process_override"
  assert projected["HANK_RUN_ROUTE_FILE"] == env["HANK_RUN_ROUTE_FILE"]
  assert child.selection_policy.default_preset == "kimi_dev"
  assert child.selection_policy.default_preset_source == "process_override"


def _route_file(tmp_path: Path) -> Path:
  route_file = tmp_path / "run-route.json"
  route_file.write_bytes(serialize_run_route({
    "schema": "run-route/2", "preset": "kimi_dev", "pool": ["ollama_cloud", "ollama"],
    "enforcement": "strict",
    "set_by": "MODEL-Q", "set_at": "2026-11-05T18:00:00Z", "reason": "fixture",
  }))
  return route_file


def _autonomous_child_env(gateway_env: dict[str, str], *, env_projection: tuple[str, ...] | None) -> dict[str, str]:
  return _positive_autonomous_child_env(
    gateway_env, provider="anthropic", profile="research_producer", deliver=False, env_projection=env_projection,
  )


@pytest.mark.parametrize("env_projection", [None, ("HANK_MODEL_AUTHORITY_DIR",)], ids=["unadmitted", "admitted"])
def test_child_of_a_gateway_with_a_route_takes_that_route(tmp_path: Path, env_projection: tuple[str, ...] | None) -> None:
  route_file = _route_file(tmp_path)
  gateway_env = {"HANK_MODEL_AUTHORITY_DIR": str(PACKAGED_AUTHORITY_DIR), "HANK_RUN_ROUTE_FILE": str(route_file)}
  child_env = _autonomous_child_env(gateway_env, env_projection=env_projection)
  parent = load_configured_model_authority(gateway_env).selection_policy.run_route
  child = load_configured_model_authority(child_env).selection_policy.run_route

  assert child_env["HANK_RUN_ROUTE_FILE"] == str(route_file)
  assert read_entry_run_route(child_env) == parent
  assert parent is not None and child == parent


@pytest.mark.parametrize("env_projection", [None, ("HANK_MODEL_AUTHORITY_DIR",)], ids=["unadmitted", "admitted"])
def test_child_of_a_gateway_with_no_route_is_handed_none_and_never_reads_the_operator_file(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_projection: tuple[str, ...] | None,
) -> None:
  # `:8000`: the operator file exists on this host, the gateway was handed nothing.
  operator = tmp_path / ".config" / "hank" / "run-route.json"
  operator.parent.mkdir(parents=True)
  operator.write_bytes(_route_file(tmp_path).read_bytes())
  monkeypatch.setenv("HOME", str(tmp_path))
  gateway_env = {"HANK_MODEL_AUTHORITY_DIR": str(PACKAGED_AUTHORITY_DIR)}
  child_env = _autonomous_child_env(gateway_env, env_projection=env_projection)

  assert child_env["HANK_RUN_ROUTE_FILE"] == ""
  assert read_entry_run_route(child_env) is None
  authority = load_configured_model_authority(child_env)
  assert authority.selection_policy.run_route is None
  assert authority.selection_policy.default_preset is None


def test_stdio_mcp_child_is_handed_the_gateways_route_or_none(tmp_path: Path) -> None:
  route_file = _route_file(tmp_path)
  assert build_mcp_env(None, environ={}, base_environ={})["HANK_RUN_ROUTE_FILE"] == ""
  assert build_mcp_env(None, environ={}, base_environ={"HANK_RUN_ROUTE_FILE": ""})["HANK_RUN_ROUTE_FILE"] == ""
  handed = build_mcp_env(None, environ={}, base_environ={"HANK_RUN_ROUTE_FILE": str(route_file)})
  assert handed["HANK_RUN_ROUTE_FILE"] == str(route_file)

def test_model_calling_children_inherit_the_store_binding_without_raw_keys() -> None:
  parent = {
    "USER_DATA_DIR": "/instance/user-data",
    "HANK_CREDENTIAL_POOL": "0",
    "OPENAI_API_KEY": "raw-openai-key",
    "ANTHROPIC_API_KEY": "raw-anthropic-key",
  }

  mcp = build_mcp_env(None, environ=parent, base_environ=parent)
  autonomous = filter_child_environment(parent, purpose=AUTONOMOUS_CHILD_ENVIRONMENT)

  for child in (mcp, autonomous):
    assert child["USER_DATA_DIR"] == "/instance/user-data"
    assert child["HANK_CREDENTIAL_POOL"] == "0"
    assert "OPENAI_API_KEY" not in child
    assert "ANTHROPIC_API_KEY" not in child


def test_child_inherits_the_spend_ledger_paths_and_attribution() -> None:
  names = (
    "HANK_SPEND_LEDGER_PATH",
    "HANK_SPEND_SETTINGS_PATH",
    "HANK_SPEND_REPO",
    "HANK_SPEND_SERVICE",
    "HANK_SPEND_INSTANCE_ID",
    "HANK_SPEND_GIT_SHA",
  )
  env = {name: f"value-of-{name}" for name in names}
  env["HANK_SPEND_UNLISTED"] = "must-not-be-inherited"

  projected = filter_child_environment(env, purpose=AUTONOMOUS_CHILD_ENVIRONMENT)

  assert projected == {name: f"value-of-{name}" for name in names}


def test_child_inherits_the_audit_hmac_secret() -> None:
  env = {
    "GATEWAY_AUDIT_HMAC_SECRET": "deployment-audit-secret",
    "FMS_UNDO_HMAC_SECRET": "must-not-be-inherited",
  }

  projected = filter_child_environment(env, purpose=AUTONOMOUS_CHILD_ENVIRONMENT)

  assert projected == {"GATEWAY_AUDIT_HMAC_SECRET": "deployment-audit-secret"}


def test_child_inherits_private_edgar_state_without_unlisted_settings() -> None:
  env = {
    "EDGAR_API_STATE_DIR": "/private/edgar",
    "EDGAR_API_EXPORT_DIR": "/private/edgar/exports",
    "EDGAR_TABLE_READER_CACHE_DIR": "/private/edgar/exports",
    "DOCUMENT_SERVICE_FILINGS_DIR": "/private/edgar/filings",
    "EDGAR_RAW_STORE_DIR": "/private/edgar/raw",
    "EDGAR_API_LOG_DIR": "/private/edgar/logs",
    "EDGAR_API_RUNTIME_DIR": "/private/edgar/runtime",
    "EDGAR_UNLISTED": "must-not-be-inherited",
  }

  projected = filter_child_environment(env, purpose=AUTONOMOUS_CHILD_ENVIRONMENT)

  assert projected == {
    name: value for name, value in env.items() if name != "EDGAR_UNLISTED"
  }
