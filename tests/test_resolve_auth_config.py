# ruff: noqa: E402

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import resolve_auth_config
from agent_gateway._provider_utils import _resolve_provider
from agent_gateway.auth import (
  AuthConfig,
  evaluate_placement_eligibility,
  filter_fallback_entries_for_placement,
)
from model_authority.loader import PACKAGED_AUTHORITY_DIR, load_model_authority
from model_authority.selection import project_fallback_chain


@pytest.fixture(autouse=True)
def _clear_anthropic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  # Cooldowns live in the store's selection sidecar: keep it private to the test.
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "user-data"))
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKENS", raising=False)
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_MODE", raising=False)
  monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
  monkeypatch.delenv("ALLOWED_MODELS_ANTHROPIC", raising=False)
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)
  monkeypatch.delenv("OPENAI_AUTH_TOKEN", raising=False)
  monkeypatch.delenv("XAI_API_KEY", raising=False)
  monkeypatch.delenv("XAI_AUTH_TOKEN", raising=False)


def test_resolve_auth_config_explicit_api_key_arg() -> None:
  assert resolve_auth_config(api_key="sk-123") == {
    "auth_mode": "api",
    "api_key": "sk-123",
    "auth_token": "",
  }


def test_resolve_auth_config_explicit_auth_token_arg() -> None:
  assert resolve_auth_config(auth_token="tok-123") == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok-123",
  }


def test_resolve_auth_config_both_args_api_key_wins() -> None:
  resolved = resolve_auth_config(api_key="sk-123", auth_token="tok-123")

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "sk-123"


def test_resolve_auth_config_config_api_key_infers_api_mode() -> None:
  assert resolve_auth_config(auth_config={"api_key": "cfg-key"}) == {
    "auth_mode": "api",
    "api_key": "cfg-key",
    "auth_token": "",
  }


def test_resolve_auth_config_config_auth_token_infers_oauth_mode() -> None:
  assert resolve_auth_config(auth_config={"auth_token": "cfg-token"}) == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "cfg-token",
  }


def test_resolve_auth_config_config_with_both_credentials_api_key_wins() -> None:
  resolved = resolve_auth_config(auth_config={"api_key": "cfg-key", "auth_token": "cfg-token"})

  assert resolved["auth_mode"] == "api"


def test_resolve_auth_config_explicit_auth_mode_is_used() -> None:
  resolved = resolve_auth_config(
    auth_config={
      "auth_mode": "oauth",
      "api_key": "cfg-key",
      "auth_token": "cfg-token",
    }
  )

  assert resolved["auth_mode"] == "oauth"
  assert resolved["api_key"] == "cfg-key"
  assert resolved["auth_token"] == "cfg-token"


def test_resolve_auth_config_ignores_ambient_anthropic_api_key() -> None:
  # u9 ruling (2026-10-12): an Anthropic record missing from args/config is
  # filled from the gateway pool only, never from the process environment.
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    assert resolve_auth_config() == {
      "auth_mode": "api",
      "api_key": "",
      "auth_token": "",
    }
  finally:
    monkeypatch.undo()


def test_resolve_auth_config_binds_the_pool_token() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  try:
    assert resolve_auth_config() == {
      "auth_mode": "oauth",
      "api_key": "",
      "auth_token": "env-token",
    }
  finally:
    monkeypatch.undo()


def test_resolve_auth_config_consults_pool_first_and_respects_parked_token() -> None:
  import json
  import time
  from agent_gateway.providers.anthropic_oauth import resolve_anthropic_credentials
  from agent_gateway.providers.anthropic_quota import Cooldown, record_cooldown

  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-1")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps(["token-1", "token-2"]))
  try:
    # 1. Unblocked token-1 binds first
    resolved_1 = resolve_auth_config()
    assert resolved_1["auth_token"] == "token-1"

    # 2. Block token-1; pool selection chooses unblocked sibling token-2
    parked = resolve_anthropic_credentials().by_token("token-1")
    assert parked is not None
    record_cooldown(
      parked.sidecar_store_path, parked.locator,
      Cooldown(scope="shared", until=time.time() + 3600, reason="rate_limit"),
    )
    resolved_2 = resolve_auth_config()
    assert resolved_2["auth_token"] == "token-2"

    # 3. Pool of one fallback when blocked: select binds it, avoiding upfront failure
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKENS")
    resolved_3 = resolve_auth_config()
    assert resolved_3["auth_token"] == "token-1"
  finally:
    monkeypatch.undo()


def test_resolve_auth_config_never_reads_env_auth_mode() -> None:
  # The mode follows the material that resolved, not ANTHROPIC_AUTH_MODE.
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "api")
  try:
    resolved = resolve_auth_config()
    with_config = resolve_auth_config(auth_config={"api_key": "cfg-key"})
  finally:
    monkeypatch.undo()

  assert resolved["auth_mode"] == "oauth"
  assert resolved["api_key"] == ""
  assert resolved["auth_token"] == "env-token"
  assert with_config["auth_mode"] == "api"
  assert with_config["api_key"] == "cfg-key"


def test_resolve_auth_config_precedence_arg_over_config_over_env() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    resolved = resolve_auth_config(
      api_key="arg-key",
      auth_config={"api_key": "cfg-key"},
    )
  finally:
    monkeypatch.undo()

  assert resolved["api_key"] == "arg-key"
  assert resolved["auth_mode"] == "api"


def test_resolve_auth_config_no_env_fallback_when_auth_config_provided() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    resolved = resolve_auth_config(auth_config={})
  finally:
    monkeypatch.undo()

  assert resolved == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
  }


def test_resolve_auth_config_prefix_sniffing_reclassifies_api_key_as_oauth() -> None:
  assert resolve_auth_config(
    api_key="sk-ant-oat-123",
    infer_mode_from_prefix=True,
  ) == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "sk-ant-oat-123",
  }


def test_resolve_auth_config_prefix_sniffing_is_off_by_default() -> None:
  assert resolve_auth_config(api_key="sk-ant-oat-123") == {
    "auth_mode": "api",
    "api_key": "sk-ant-oat-123",
    "auth_token": "",
  }


def test_resolve_auth_config_explicit_auth_mode_beats_prefix_sniffing() -> None:
  resolved = resolve_auth_config(
    api_key="sk-ant-oat-123",
    auth_config={"auth_mode": "api"},
    infer_mode_from_prefix=True,
  )

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "sk-ant-oat-123"
  assert resolved["auth_token"] == ""


def test_resolve_auth_config_no_credentials_without_raise_returns_empty_api_mode() -> None:
  assert resolve_auth_config() == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
  }


def test_resolve_auth_config_no_credentials_with_raise_errors() -> None:
  with pytest.raises(RuntimeError, match="auth_mode is 'api' but no api_key found"):
    resolve_auth_config(raise_on_missing=True)


def test_resolve_auth_config_openai_explicit_api_key_includes_auth_fields() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    api_key="sk-openai",
  )

  assert resolved == {
    "auth_mode": "api",
    "api_key": "sk-openai",
    "auth_token": "",
  }


@pytest.mark.parametrize(
  ("provider", "env_name"),
  [
    ("openai", "OPENAI_API_KEY"),
    ("xai", "XAI_API_KEY"),
  ],
)
def test_resolve_auth_config_snapshots_non_anthropic_env_api_key(
  monkeypatch: pytest.MonkeyPatch,
  provider: str,
  env_name: str,
) -> None:
  monkeypatch.setenv(env_name, "env-service-key")

  resolved = resolve_auth_config(provider=provider)
  monkeypatch.delenv(env_name)

  assert resolved == {
    "auth_mode": "api",
    "api_key": "env-service-key",
    "auth_token": "",
  }


def test_resolve_auth_config_openai_explicit_auth_token() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    auth_token="oat-xxx",
  )

  assert resolved == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "oat-xxx",
  }


def test_resolve_auth_config_openai_api_key_wins_over_auth_token() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    api_key="sk-openai",
    auth_token="oat-xxx",
  )

  assert resolved == {
    "auth_mode": "api",
    "api_key": "sk-openai",
    "auth_token": "",
  }


def test_resolve_auth_config_openai_auth_config_with_explicit_mode() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    auth_config={
      "auth_mode": "oauth",
      "auth_token": "tok",
    },
  )

  assert resolved == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
  }


def test_resolve_auth_config_openai_no_credentials_returns_minimal() -> None:
  resolved = resolve_auth_config(provider="openai")

  assert "auth_mode" not in resolved
  assert "auth_token" not in resolved
  assert "api_key" not in resolved


def test_resolve_auth_config_openai_whitespace_token_ignored() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    auth_token="   ",
  )

  assert "auth_mode" not in resolved
  assert "auth_token" not in resolved
  assert "api_key" not in resolved


def test_resolve_auth_config_openai_auth_config_infers_mode_from_token() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    auth_config={"auth_token": "tok"},
  )

  assert resolved == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
  }


def test_resolve_auth_config_sets_request_limit_without_selection() -> None:
  resolved = resolve_auth_config(
    api_key="sk-123",
    max_tokens=123,
  )

  assert resolved["max_tokens"] == 123
  assert {
    "model",
    "model_key",
    "effort",
    "thinking",
    "thinking_enabled_requested",
  }.isdisjoint(resolved)


@pytest.mark.parametrize(
  "field_name",
  [
    "model",
    "model_key",
    "effort",
    "thinking",
    "thinking_enabled_requested",
  ],
)
def test_resolve_auth_config_rejects_selection_in_auth_config(
  field_name: str,
) -> None:
  with pytest.raises(ValueError, match="not model-selection authority"):
    resolve_auth_config(auth_config={field_name: "forbidden"})


def test_resolve_auth_config_does_not_inject_model_defaults() -> None:
  resolved = resolve_auth_config(api_key="sk-123")

  assert "model" not in resolved
  assert "max_tokens" not in resolved
  assert "thinking" not in resolved


def test_resolve_auth_config_blank_args_fall_through_to_next_tier() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  try:
    resolved = resolve_auth_config(api_key="   ", auth_token="")
  finally:
    monkeypatch.undo()

  assert resolved["auth_mode"] == "oauth"
  assert resolved["api_key"] == ""
  assert resolved["auth_token"] == "env-token"


def test_resolve_auth_config_preserves_extra_auth_config_keys() -> None:
  resolved = resolve_auth_config(
    auth_config={
      "api_key": "cfg-key",
      "custom_field": 42,
      "nested": {"enabled": True},
    }
  )

  assert resolved["custom_field"] == 42
  assert resolved["nested"] == {"enabled": True}
  assert resolved["api_key"] == "cfg-key"
  assert resolved["auth_mode"] == "api"


def test_resolve_auth_config_read_env_false_skips_the_pool() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  try:
    resolved = resolve_auth_config(read_env=False)
  finally:
    monkeypatch.undo()

  assert resolved == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
  }


def test_resolve_auth_config_preserves_non_active_credential_with_auth_config() -> None:
  resolved = resolve_auth_config(auth_config={"api_key": "cfg-key", "auth_token": "cfg-token"})

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "cfg-key"
  assert resolved["auth_token"] == "cfg-token"


def test_resolve_auth_config_blanks_non_active_credential_without_auth_config() -> None:
  resolved = resolve_auth_config(api_key="arg-key", auth_token="arg-token")

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "arg-key"
  assert resolved["auth_token"] == ""


def test_resolve_provider_binds_anthropic_from_the_pool_only() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "api")
  try:
    _provider, _provider_name, config = _resolve_provider(
      "anthropic", "claude-opus-5", None, None, None
    )
  finally:
    monkeypatch.undo()

  assert config["auth_mode"] == "oauth"
  assert config["api_key"] == ""
  assert config["auth_token"] == "env-token"


def test_resolve_provider_requires_bound_upstream_model() -> None:
  with pytest.raises(
    ValueError,
    match="requires an upstream model from a CapabilityBind",
  ):
    _resolve_provider(
      "anthropic",
      None,
      None,
      None,
      None,
      auth_config={"api_key": "cfg-key"},
    )


def test_resolve_provider_keeps_explicit_model_out_of_auth_material() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "anthropic",
    "opus",
    None,
    None,
    None,
    auth_config={"api_key": "cfg-key"},
  )

  assert "model" not in config
  assert config["auth_mode"] == "api"
  assert config["api_key"] == "cfg-key"


def test_resolve_provider_uses_auth_config_credentials_instead_of_env() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    _provider, _provider_name, config = _resolve_provider(
      "anthropic",
      "claude-opus-5",
      None,
      None,
      None,
      auth_config={"api_key": "cfg-key"},
    )
  finally:
    monkeypatch.undo()

  assert config["api_key"] == "cfg-key"
  assert config["auth_mode"] == "api"


_AUTH_MECHANISMS = ("api_key", "oauth_bearer", "none")
_ENTITLEMENT_KINDS = (
  "commercial_byok",
  "consumer_subscription",
  "org_commercial",
  "operator_personal_subscription",
  "platform_metered",
  "local_unauthenticated",
)
_HOSTED_REJECTED_ROUTES = (
  ("ollama", "ollama.local"),
  ("ollama", "ollama.daemon_cloud"),
  ("codex", "codex.chatgpt"),
  ("google_antigravity", "google.antigravity"),
  ("copilot", "copilot.api"),
)
_PLACEMENT_SLOTS = (
  ("session.driver", "default"),
  ("node.implement", "build"),
  ("node.explore", "fast"),
  ("investment.research_agent", "standard"),
)
_STRIPPED_HOSTED_ROUTES = frozenset({
  "codex.chatgpt",
  "google.antigravity",
  "ollama.local",
  "ollama.daemon_cloud",
})


def test_auth_config_separates_auth_mechanism_and_entitlement_kind() -> None:
  for mechanism in _AUTH_MECHANISMS:
    for kind in _ENTITLEMENT_KINDS:
      config = AuthConfig.from_dict({
        "provider": "anthropic",
        "billing_mode": "byok" if kind in {"commercial_byok", "consumer_subscription"} else "metered",
        "auth_mechanism": mechanism,
        "entitlement_kind": kind,
        "api_key": "sk-ant-api03-independent",
        "auth_mode": "oauth" if mechanism == "oauth_bearer" else "api",
      })
      assert config.auth_mechanism == mechanism
      assert config.entitlement_kind == kind
      materialized = config.to_dict()
      assert materialized["auth_mechanism"] == mechanism
      assert materialized["entitlement_kind"] == kind

  # A Muse device-flow subscription is a key on the wire and a consumer seat
  # in billing. auth_mode=oauth must not collapse either axis.
  muse_raw = {
    "provider": "muse",
    "billing_mode": "byok",
    "api_key": "muse-subscription-key",
    "auth_mode": "oauth",
    "source": "subscription",
    "auth_store_path": "/data/users/1/muse/credentials.json",
  }
  muse = AuthConfig.from_dict(muse_raw)
  assert muse.auth_mechanism == "api_key"
  assert muse.entitlement_kind == "consumer_subscription"
  assert muse.to_dict() == muse_raw

  local = AuthConfig.from_dict({"provider": "ollama", "billing_mode": "metered"})
  assert local.auth_mechanism == "none"
  assert local.entitlement_kind == "local_unauthenticated"


def _placement(provider: str, route: str, kind: str, placement: str):
  return evaluate_placement_eligibility(
    provider=provider, route=route, entitlement_kind=kind, placement=placement,
  )


def test_evaluate_placement_eligibility_rules() -> None:
  for kind in ("consumer_subscription", "operator_personal_subscription"):
    rejected = _placement("anthropic", "anthropic.public", kind, "hosted_external")
    assert rejected.eligible is False
    assert rejected.credential_principal is None
    assert rejected.billing_mode is None

  for provider, route in _HOSTED_REJECTED_ROUTES:
    rejected = _placement(provider, route, "commercial_byok", "hosted_external")
    assert rejected.eligible is False, route
    assert (rejected.credential_principal, rejected.billing_mode) == (None, None)

  hosted_byok = _placement("anthropic", "anthropic.public", "commercial_byok", "hosted_external")
  assert hosted_byok.eligible is True
  assert (hosted_byok.credential_principal, hosted_byok.billing_mode) == ("user", "byok")
  hosted_muse = _placement("muse", "muse.api", "commercial_byok", "hosted_external")
  assert (hosted_muse.credential_principal, hosted_muse.billing_mode) == ("user", "byok")
  for kind in ("org_commercial", "platform_metered"):
    admitted = _placement("anthropic", "anthropic.public", kind, "hosted_external")
    assert admitted.eligible is True
    assert (admitted.credential_principal, admitted.billing_mode) == ("service", "metered")

  for kind in ("consumer_subscription", "commercial_byok"):
    admitted = _placement("muse", "muse.api", kind, "local")
    assert admitted.eligible is True
    assert (admitted.credential_principal, admitted.billing_mode) == ("user", "byok")
  for kind in ("operator_personal_subscription", "org_commercial"):
    admitted = _placement("anthropic", "anthropic.public", kind, "local")
    assert admitted.eligible is True
    assert (admitted.credential_principal, admitted.billing_mode) == ("service", "metered")


def _routes(entries, registry) -> set[str]:
  return {
    registry.models[entry.model_key].route
    for entry in entries
    if getattr(entry, "model_key", None) in registry.models
  }


def test_fallback_chain_filtering_under_hosted_external() -> None:
  authority = load_model_authority(PACKAGED_AUTHORITY_DIR, local_fallback_enabled=True)
  registry = authority.registry
  projected_routes: set[str] = set()
  for capability, role in _PLACEMENT_SLOTS:
    entries = project_fallback_chain(
      registry, authority.selection_policy, capability, role,
    ).entries
    routes = _routes(entries, registry)
    projected_routes |= routes

    stripped = filter_fallback_entries_for_placement(
      entries, registry, placement="hosted_external",
      entitlement_by_provider={"muse": "consumer_subscription", "anthropic": "commercial_byok"},
    )
    stripped_routes = _routes(stripped, registry)
    assert stripped_routes.isdisjoint(_STRIPPED_HOSTED_ROUTES)
    assert "muse.api" not in stripped_routes
    if "anthropic.public" in routes:
      assert "anthropic.public" in stripped_routes

    for muse_kind in ("commercial_byok", "org_commercial"):
      kept = filter_fallback_entries_for_placement(
        entries, registry, placement="hosted_external",
        entitlement_by_provider={"muse": muse_kind, "anthropic": muse_kind},
      )
      kept_routes = _routes(kept, registry)
      assert kept_routes.isdisjoint(_STRIPPED_HOSTED_ROUTES)
      if "muse.api" in routes:
        assert "muse.api" in kept_routes
      if "anthropic.public" in routes:
        assert "anthropic.public" in kept_routes

  assert _STRIPPED_HOSTED_ROUTES <= projected_routes
  assert {"muse.api", "anthropic.public"} <= projected_routes
