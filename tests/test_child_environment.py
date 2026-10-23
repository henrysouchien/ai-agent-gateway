import pytest

from model_authority.loader import PACKAGED_AUTHORITY_DIR, load_configured_model_authority
from model_authority.selection import project_fallback_chain

from agent_gateway.child_environment import (
  AUTONOMOUS_CHILD_ENVIRONMENT,
  AUTONOMOUS_CHILD_PROFILES,
  filter_child_environment,
)


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


def test_child_inherits_the_parents_default_preset() -> None:
  env = {
    "HANK_MODEL_AUTHORITY_DIR": str(PACKAGED_AUTHORITY_DIR),
    "HANK_MODEL_PRESET": "kimi_dev",
  }
  parent = load_configured_model_authority(env)
  projected = filter_child_environment(env, purpose=AUTONOMOUS_CHILD_ENVIRONMENT)
  child = load_configured_model_authority(projected)

  assert parent.selection_policy.default_preset == "kimi_dev"
  assert "HANK_MODEL_PRESET" in projected
  assert child.selection_policy.default_preset == "kimi_dev"


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
