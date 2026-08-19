"""Deployment-selected registry/policy artifacts (review items C1 and C4).

Covers: config-only model addition through an alternative admitted artifact,
explicit lifecycle authorability with coherence rejection, loud construction
failure on invalid/unknown-field artifacts, deployment selection via the
documented environment variables, and behavioral equivalence of the packaged
artifacts with the frozen product-decision inventory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from agent_gateway.capability_binding import (
  AuthContext,
  CredentialHandle,
  ModelSelectionIntent,
  resolve_capability_model,
)
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  DEFAULT_MODEL_REGISTRY_ARTIFACT,
  DEFAULT_MODEL_SELECTION_ARTIFACT,
  GATEWAY_EXECUTED_CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
  MODEL_REGISTRY_FILE_ENV_VAR,
  MODEL_SELECTION_FILE_ENV_VAR,
  load_model_registry,
  load_model_selection_policy,
)
from agent_gateway.providers import installed_adapter_route_support
from agent_gateway.server import GatewayServerConfig, create_gateway_app

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

_NEW_MODEL_KEY = "anthropic.claude-nova-6"
_NEW_MODEL_ENTRY = {
  "key": _NEW_MODEL_KEY,
  "label": "Nova 6",
  "provider": "anthropic",
  "upstream_model": "claude-nova-6",
  "adapter": "anthropic.messages",
  "protocol_profile": "messages.adaptive",
  "route": "anthropic.public",
  "lifecycle": "active",
  "capabilities": {
    "session.driver": "user_selectable",
    "plan.author": "internal",
    "node.fork": "internal",
  },
  "supported_efforts": ["low", "medium", "high"],
  "default_effort": "high",
  "features": ["tools", "streaming"],
  "reported_identities": ["claude-nova-6"],
}


def _registry_document() -> dict:
  return yaml.safe_load(DEFAULT_MODEL_REGISTRY_ARTIFACT.read_text(encoding="utf-8"))


def _selection_document() -> dict:
  return yaml.safe_load(DEFAULT_MODEL_SELECTION_ARTIFACT.read_text(encoding="utf-8"))


def _write_artifact(path: Path, document: dict) -> Path:
  path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
  return path


def _registry_with_entry(tmp_path: Path, entry: dict) -> Path:
  document = _registry_document()
  document["revision"] = "2026-08-14.test"
  document["models"].append(entry)
  return _write_artifact(tmp_path / "registry.yaml", document)


def _selection_allowing_new_driver(tmp_path: Path) -> Path:
  document = _selection_document()
  document["revision"] = "2026-08-14.test"
  document["capabilities"]["session.driver"]["allowed_model_keys"].append(
    _NEW_MODEL_KEY
  )
  return _write_artifact(tmp_path / "selection.yaml", document)


def _auth(model_keys: frozenset[str]) -> AuthContext:
  return AuthContext(
    run_mode="interactive",
    actor_id="user-7",
    tenant_id="tenant-1",
    user_provider_handles={
      "anthropic": CredentialHandle(
        handle_id="credential:anthropic:user-7",
        provider="anthropic",
        principal="user",
        tenant_id="tenant-1",
        actor_id="user-7",
      ),
    },
    service_provider_handles={},
    entitled_capabilities=frozenset({"session.driver"}),
    entitled_model_keys=model_keys,
  )


# --- Behavioral equivalence with the frozen literals (product decisions §6) ---


_EXPECTED_EXECUTION_IDENTITIES = {
  "anthropic.claude-fable-5": ("anthropic", "claude-fable-5", "anthropic.messages", "messages.adaptive", "anthropic.public", "active", "high"),
  "anthropic.claude-haiku-4-5": ("anthropic", "claude-haiku-4-5", "anthropic.messages", "messages.standard", "anthropic.public", "active", "none"),
  "anthropic.claude-mythos-5": ("anthropic", "claude-mythos-5", "anthropic.messages", "messages.adaptive", "anthropic.public", "active", "high"),
  "anthropic.claude-opus-5": ("anthropic", "claude-opus-5", "anthropic.messages", "messages.adaptive", "anthropic.public", "active", "high"),
  "anthropic.claude-sonnet-5": ("anthropic", "claude-sonnet-5", "anthropic.messages", "messages.adaptive", "anthropic.public", "active", "high"),
  "openai.gpt-5-6": ("openai", "gpt-5.6", "openai.responses", "responses.reasoning", "openai.public", "active", "medium"),
  "codex.gpt-5-6-luna": ("codex", "gpt-5.6-luna", "codex.responses", "codex.reasoning", "codex.chatgpt", "active", "medium"),
  "codex.gpt-5-6-sol": ("codex", "gpt-5.6-sol", "codex.responses", "codex.reasoning", "codex.chatgpt", "active", "medium"),
  "codex.gpt-5-6-terra": ("codex", "gpt-5.6-terra", "codex.responses", "codex.reasoning", "codex.chatgpt", "active", "medium"),
  "xai.grok-4-5": ("xai", "grok-4.5", "xai.responses", "responses.reasoning", "xai.public", "active", "medium"),
  "anthropic.claude-sonnet-4-6-sdk": ("anthropic", "claude-sonnet-4-6", "anthropic.sdk.messages", "messages.standard", "anthropic.byok", "hidden", "none"),
  "anthropic.claude-haiku-4-5-20251001-sdk": ("anthropic", "claude-haiku-4-5-20251001", "anthropic.sdk.messages", "messages.standard", "anthropic.byok", "hidden", "none"),
  "anthropic.claude-haiku-4-5-20251001-gateway": ("anthropic", "claude-haiku-4-5-20251001", "anthropic.messages", "messages.standard", "anthropic.public", "hidden", "none"),
  "anthropic.claude-opus-4-8-oauth": ("anthropic", "claude-opus-4-8", "anthropic.sdk.messages", "messages.oauth", "anthropic.oauth", "hidden", "none"),
  "anthropic.claude-sonnet-4-20250514-sdk": ("anthropic", "claude-sonnet-4-20250514", "anthropic.sdk.messages", "messages.standard", "anthropic.service", "hidden", "none"),
  "openai.gpt-5-4-mini-sdk": ("openai", "gpt-5.4-mini", "openai.sdk.chat_completions", "chat_completions.standard", "openai.service", "hidden", "none"),
}

_EXPECTED_DEFAULTS = {
  "session.driver": ("model", "anthropic.claude-opus-5", "high"),
  "plan.author": ("inherit_parent", None, None),
  "node.explore": ("model", "anthropic.claude-opus-5", "high"),
  "node.implement": ("model", "anthropic.claude-opus-5", "high"),
  "node.mutate": ("model", "anthropic.claude-opus-5", "high"),
  "node.fork": ("inherit_parent", None, None),
  "node.verify": ("model", "anthropic.claude-opus-5", "high"),
  "node.choose": ("model", "anthropic.claude-opus-5", "high"),
  "citation.review": ("model", "anthropic.claude-haiku-4-5", "none"),
  "risk.completion": ("model", "anthropic.claude-sonnet-4-6-sdk", "none"),
  "risk.interpretation": ("model", "anthropic.claude-sonnet-4-6-sdk", "none"),
  "risk.peer_generation": ("model", "openai.gpt-5-4-mini-sdk", "none"),
  "risk.asset_classification": ("model", "anthropic.claude-haiku-4-5-20251001-sdk", "none"),
  "risk.overview_editorial": ("model", "anthropic.claude-haiku-4-5-20251001-sdk", "none"),
  "risk.document_ingest": ("model", "anthropic.claude-opus-4-8-oauth", "none"),
  "investment.research_agent": ("model", "anthropic.claude-sonnet-4-6-sdk", "none"),
  "investment.quant_worker": ("model", "openai.gpt-5-6", "high"),
  "investment.newsletter": ("model", "anthropic.claude-haiku-4-5-20251001-gateway", "none"),
  "investment.earnings_transcript": ("model", "anthropic.claude-haiku-4-5-20251001-gateway", "none"),
  "investment.biotech_review": ("model", "anthropic.claude-sonnet-4-20250514-sdk", "none"),
}

_DRIVER_KEYS = frozenset({
  "anthropic.claude-fable-5",
  "anthropic.claude-haiku-4-5",
  "anthropic.claude-mythos-5",
  "anthropic.claude-opus-5",
  "anthropic.claude-sonnet-5",
  "openai.gpt-5-6",
  "codex.gpt-5-6-luna",
  "codex.gpt-5-6-sol",
  "codex.gpt-5-6-terra",
  "xai.grok-4-5",
})


def test_packaged_registry_artifact_matches_frozen_inventory() -> None:
  assert INITIAL_MODEL_REGISTRY.revision == "2026-08-18.1"
  observed = {
    key: (
      entry.provider,
      entry.upstream_model,
      entry.adapter,
      entry.protocol_profile,
      entry.route,
      entry.lifecycle,
      entry.default_effort,
    )
    for key, entry in INITIAL_MODEL_REGISTRY.models.items()
  }
  assert observed == _EXPECTED_EXECUTION_IDENTITIES
  assert {
    key
    for key, entry in INITIAL_MODEL_REGISTRY.models.items()
    if entry.capabilities.get("session.driver") == "user_selectable"
  } == _DRIVER_KEYS
  haiku = INITIAL_MODEL_REGISTRY.require("anthropic.claude-haiku-4-5")
  assert haiku.reported_identities == frozenset({
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
  })
  assert INITIAL_MODEL_REGISTRY.require(
    "openai.gpt-5-6"
  ).reported_identities == frozenset({"gpt-5.6", "gpt-5.6-sol"})
  oauth = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-4-8-oauth")
  assert oauth.features == frozenset({"vision"})
  assert INITIAL_MODEL_REGISTRY.require("xai.grok-4-5").supported_efforts == (
    frozenset({"low", "medium", "high"})
  )
  assert INITIAL_MODEL_REGISTRY.require(
    "anthropic.claude-fable-5"
  ).supported_efforts == frozenset({"low", "medium", "high", "xhigh", "max"})


def test_packaged_selection_artifact_matches_frozen_policy() -> None:
  assert INITIAL_MODEL_SELECTION_POLICY.revision == "2026-08-18.1"
  assert set(INITIAL_MODEL_SELECTION_POLICY.capabilities) == CAPABILITY_IDS
  observed = {
    capability_id: (
      policy.default.kind,
      policy.default.model_key,
      policy.default.effort,
    )
    for capability_id, policy in INITIAL_MODEL_SELECTION_POLICY.capabilities.items()
  }
  assert observed == _EXPECTED_DEFAULTS
  driver = INITIAL_MODEL_SELECTION_POLICY.capabilities["session.driver"]
  assert driver.allowed_model_keys == _DRIVER_KEYS
  assert driver.allow_saved_preference
  assert driver.allow_explicit_user
  author = INITIAL_MODEL_SELECTION_POLICY.capabilities["plan.author"]
  assert author.allowed_model_keys == _DRIVER_KEYS
  assert author.allow_authenticated_run_override
  assert INITIAL_MODEL_SELECTION_POLICY.capabilities[
    "node.fork"
  ].allowed_model_keys == _DRIVER_KEYS
  review = INITIAL_MODEL_SELECTION_POLICY.capabilities["citation.review"]
  assert review.allowed_model_keys == frozenset({
    "anthropic.claude-fable-5",
    "anthropic.claude-haiku-4-5",
    "anthropic.claude-mythos-5",
    "anthropic.claude-opus-5",
    "anthropic.claude-sonnet-5",
  })
  assert review.allow_explicit_user is False
  assert review.allow_saved_preference is False
  assert review.allow_authenticated_run_override is False
  node_keys = frozenset({
    "anthropic.claude-fable-5",
    "anthropic.claude-mythos-5",
    "anthropic.claude-opus-5",
    "anthropic.claude-sonnet-5",
  })
  for capability_id in (
    "node.explore",
    "node.implement",
    "node.mutate",
    "node.verify",
    "node.choose",
  ):
    assert INITIAL_MODEL_SELECTION_POLICY.capabilities[
      capability_id
    ].allowed_model_keys == node_keys
  assert all(
    not policy.by_channel
    for policy in INITIAL_MODEL_SELECTION_POLICY.capabilities.values()
  )


def test_loading_the_packaged_artifacts_reproduces_the_shipped_authorities() -> None:
  registry = load_model_registry(DEFAULT_MODEL_REGISTRY_ARTIFACT)
  policy = load_model_selection_policy(DEFAULT_MODEL_SELECTION_ARTIFACT)
  registry.admit_adapter_support(
    installed_adapter_route_support(),
    executed_capability_ids=GATEWAY_EXECUTED_CAPABILITY_IDS,
  )
  policy.admit_registry(registry)
  assert registry == INITIAL_MODEL_REGISTRY
  assert dict(registry.models) == dict(INITIAL_MODEL_REGISTRY.models)
  assert policy == INITIAL_MODEL_SELECTION_POLICY
  assert dict(policy.capabilities) == dict(
    INITIAL_MODEL_SELECTION_POLICY.capabilities
  )


# --- Config-only model addition (plan §8 / design acceptance) ---


def test_new_model_becomes_selectable_through_configuration_alone(
  tmp_path: Path,
) -> None:
  registry = load_model_registry(_registry_with_entry(tmp_path, dict(_NEW_MODEL_ENTRY)))
  policy = load_model_selection_policy(_selection_allowing_new_driver(tmp_path))

  registry.admit_adapter_support(
    installed_adapter_route_support(),
    executed_capability_ids=GATEWAY_EXECUTED_CAPABILITY_IDS,
  )
  policy.admit_registry(registry)

  bind = resolve_capability_model(
    "session.driver",
    registry=registry,
    selection_policy=policy,
    auth=_auth(frozenset({_NEW_MODEL_KEY})),
    explicit_intent=ModelSelectionIntent(
      model_key=_NEW_MODEL_KEY,
      effort="medium",
      source="explicit_user",
    ),
  )

  assert bind.model_key == _NEW_MODEL_KEY
  assert bind.upstream_model == "claude-nova-6"
  assert bind.adapter == "anthropic.messages"
  assert bind.effort == "medium"
  assert bind.registry_revision == "2026-08-14.test"


def _run_module_import(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
  env = dict(os.environ)
  env.pop(MODEL_REGISTRY_FILE_ENV_VAR, None)
  env.pop(MODEL_SELECTION_FILE_ENV_VAR, None)
  env.update(env_overrides)
  env["PYTHONPATH"] = str(PACKAGE_ROOT)
  return subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "from agent_gateway.model_registry import INITIAL_MODEL_REGISTRY, "
        "INITIAL_MODEL_SELECTION_POLICY\n"
        "print(','.join(sorted(INITIAL_MODEL_REGISTRY.models)))\n"
        "print(','.join(sorted(INITIAL_MODEL_SELECTION_POLICY.capabilities["
        "'session.driver'].allowed_model_keys)))\n"
      ),
    ],
    cwd=PACKAGE_ROOT,
    env=env,
    capture_output=True,
    text=True,
    timeout=120,
  )


def test_deployment_env_var_selects_alternative_admitted_artifacts(
  tmp_path: Path,
) -> None:
  registry_path = _registry_with_entry(tmp_path, dict(_NEW_MODEL_ENTRY))
  selection_path = _selection_allowing_new_driver(tmp_path)

  result = _run_module_import({
    MODEL_REGISTRY_FILE_ENV_VAR: str(registry_path),
    MODEL_SELECTION_FILE_ENV_VAR: str(selection_path),
  })

  assert result.returncode == 0, result.stderr
  model_keys, allowed_driver_keys = result.stdout.strip().splitlines()
  assert _NEW_MODEL_KEY in model_keys.split(",")
  assert _NEW_MODEL_KEY in allowed_driver_keys.split(",")


def test_deployment_env_var_fails_startup_on_missing_artifact(
  tmp_path: Path,
) -> None:
  result = _run_module_import({
    MODEL_REGISTRY_FILE_ENV_VAR: str(tmp_path / "absent.yaml"),
  })
  assert result.returncode != 0
  assert "model authority artifact is unreadable" in result.stderr


def test_deployment_env_var_fails_startup_on_incoherent_artifact(
  tmp_path: Path,
) -> None:
  document = _registry_document()
  document["models"][0]["lifecycle"] = "revoked"
  artifact = _write_artifact(tmp_path / "registry.yaml", document)

  result = _run_module_import({MODEL_REGISTRY_FILE_ENV_VAR: str(artifact)})

  assert result.returncode != 0
  assert "cannot be user-selectable" in result.stderr


# --- Lifecycle authorability and coherence (C4) ---


@pytest.mark.parametrize("lifecycle", ["deprecated", "disabled", "revoked"])
def test_all_lifecycle_states_are_authorable_for_internal_models(
  tmp_path: Path,
  lifecycle: str,
) -> None:
  document = _registry_document()
  target = next(
    entry
    for entry in document["models"]
    if entry["key"] == "anthropic.claude-sonnet-4-20250514-sdk"
  )
  target["lifecycle"] = lifecycle
  registry = load_model_registry(_write_artifact(tmp_path / "registry.yaml", document))

  assert registry.require("anthropic.claude-sonnet-4-20250514-sdk").lifecycle == (
    lifecycle
  )


@pytest.mark.parametrize("lifecycle", ["deprecated", "disabled", "revoked", "hidden"])
def test_user_selectable_entry_cannot_author_non_active_lifecycle(
  tmp_path: Path,
  lifecycle: str,
) -> None:
  document = _registry_document()
  document["models"][0]["lifecycle"] = lifecycle

  with pytest.raises(ValueError, match="cannot be user-selectable"):
    load_model_registry(_write_artifact(tmp_path / "registry.yaml", document))


@pytest.mark.parametrize("lifecycle", ["deprecated", "disabled", "revoked"])
def test_policy_admission_rejects_retired_model_for_new_selection(
  tmp_path: Path,
  lifecycle: str,
) -> None:
  document = _registry_document()
  target = next(
    entry
    for entry in document["models"]
    if entry["key"] == "anthropic.claude-sonnet-4-20250514-sdk"
  )
  target["lifecycle"] = lifecycle
  registry = load_model_registry(_write_artifact(tmp_path / "registry.yaml", document))
  policy = load_model_selection_policy(DEFAULT_MODEL_SELECTION_ARTIFACT)

  with pytest.raises(ValueError, match=f"allows {lifecycle} model"):
    policy.admit_registry(registry)


def test_unknown_lifecycle_value_is_rejected(tmp_path: Path) -> None:
  document = _registry_document()
  target = next(
    entry
    for entry in document["models"]
    if entry["key"] == "anthropic.claude-sonnet-4-20250514-sdk"
  )
  target["lifecycle"] = "sunset"

  with pytest.raises(ValueError, match="unknown .*lifecycle"):
    load_model_registry(_write_artifact(tmp_path / "registry.yaml", document))


# --- Loud failure on invalid artifacts ---


def test_unknown_registry_entry_field_is_rejected(tmp_path: Path) -> None:
  document = _registry_document()
  document["models"][0]["pricing_tier"] = "premium"

  with pytest.raises(ValueError, match="unknown fields: pricing_tier"):
    load_model_registry(_write_artifact(tmp_path / "registry.yaml", document))


def test_unknown_registry_document_field_is_rejected(tmp_path: Path) -> None:
  document = _registry_document()
  document["fallback_model"] = "anthropic.claude-opus-5"

  with pytest.raises(ValueError, match="unknown fields: fallback_model"):
    load_model_registry(_write_artifact(tmp_path / "registry.yaml", document))


def test_missing_registry_entry_field_is_rejected(tmp_path: Path) -> None:
  document = _registry_document()
  del document["models"][0]["reported_identities"]

  with pytest.raises(ValueError, match="missing fields: reported_identities"):
    load_model_registry(_write_artifact(tmp_path / "registry.yaml", document))


def test_unknown_selection_policy_field_is_rejected(tmp_path: Path) -> None:
  document = _selection_document()
  document["capabilities"]["session.driver"]["fallback_model_key"] = (
    "anthropic.claude-sonnet-5"
  )

  with pytest.raises(ValueError, match="unknown fields: fallback_model_key"):
    load_model_selection_policy(_write_artifact(tmp_path / "selection.yaml", document))


def test_unknown_selection_default_field_is_rejected(tmp_path: Path) -> None:
  document = _selection_document()
  document["capabilities"]["session.driver"]["default"]["provider"] = "anthropic"

  with pytest.raises(ValueError, match="unknown fields: provider"):
    load_model_selection_policy(_write_artifact(tmp_path / "selection.yaml", document))


def test_incomplete_capability_set_is_rejected(tmp_path: Path) -> None:
  document = _selection_document()
  del document["capabilities"]["risk.completion"]

  with pytest.raises(ValueError, match="complete capability set"):
    load_model_selection_policy(_write_artifact(tmp_path / "selection.yaml", document))


def test_non_yaml_artifact_is_rejected(tmp_path: Path) -> None:
  artifact = tmp_path / "registry.yaml"
  artifact.write_text("models: [unclosed", encoding="utf-8")

  with pytest.raises(ValueError, match="invalid YAML"):
    load_model_registry(artifact)


def test_missing_artifact_file_is_rejected(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="unreadable"):
    load_model_registry(tmp_path / "absent.yaml")


def test_duplicate_model_key_is_rejected(tmp_path: Path) -> None:
  document = _registry_document()
  document["models"].append(dict(document["models"][0]))

  with pytest.raises(ValueError, match="duplicate model key"):
    load_model_registry(_write_artifact(tmp_path / "registry.yaml", document))


# --- Duplicate YAML mapping keys fail construction loudly ---
#
# The default SafeLoader keeps the last value for a repeated mapping key, which
# would let an artifact silently replace an authored value.  These tests author
# duplicates as raw YAML text (a dict round-trip cannot represent them).


def test_duplicate_top_level_key_in_registry_artifact_is_rejected(
  tmp_path: Path,
) -> None:
  raw = DEFAULT_MODEL_REGISTRY_ARTIFACT.read_text(encoding="utf-8")
  artifact = tmp_path / "registry.yaml"
  artifact.write_text(raw + '\nrevision: "9999-01-01.override"\n', encoding="utf-8")

  with pytest.raises(ValueError, match="duplicate mapping key: 'revision'"):
    load_model_registry(artifact)


def test_duplicate_nested_key_in_registry_artifact_is_rejected(
  tmp_path: Path,
) -> None:
  raw = DEFAULT_MODEL_REGISTRY_ARTIFACT.read_text(encoding="utf-8")
  needle = '    label: "Fable 5"'
  assert needle in raw
  raw = raw.replace(needle, needle + '\n    label: "Fable 5 override"', 1)
  artifact = tmp_path / "registry.yaml"
  artifact.write_text(raw, encoding="utf-8")

  with pytest.raises(ValueError, match="duplicate mapping key: 'label'"):
    load_model_registry(artifact)


def test_duplicate_top_level_key_in_selection_artifact_is_rejected(
  tmp_path: Path,
) -> None:
  raw = DEFAULT_MODEL_SELECTION_ARTIFACT.read_text(encoding="utf-8")
  artifact = tmp_path / "selection.yaml"
  artifact.write_text(
    raw + '\nschema: "product-model-selection/v1"\n', encoding="utf-8"
  )

  with pytest.raises(ValueError, match="duplicate mapping key: 'schema'"):
    load_model_selection_policy(artifact)


def test_duplicate_capability_block_in_selection_artifact_is_rejected(
  tmp_path: Path,
) -> None:
  raw = DEFAULT_MODEL_SELECTION_ARTIFACT.read_text(encoding="utf-8")
  duplicate_block = (
    "\n"
    "  session.driver:\n"
    '    default: {kind: "model", model_key: "anthropic.claude-sonnet-5", '
    'effort: "high"}\n'
    "    by_channel: {}\n"
    '    allowed_model_keys: ["anthropic.claude-sonnet-5"]\n'
  )
  artifact = tmp_path / "selection.yaml"
  artifact.write_text(raw + duplicate_block, encoding="utf-8")

  with pytest.raises(ValueError, match="duplicate mapping key: 'session.driver'"):
    load_model_selection_policy(artifact)


def test_duplicate_key_in_flow_mapping_default_is_rejected(tmp_path: Path) -> None:
  raw = DEFAULT_MODEL_SELECTION_ARTIFACT.read_text(encoding="utf-8")
  needle = '{kind: "inherit_parent"}'
  assert needle in raw
  raw = raw.replace(
    needle,
    '{kind: "inherit_parent", kind: "model"}',
    1,
  )
  artifact = tmp_path / "selection.yaml"
  artifact.write_text(raw, encoding="utf-8")

  with pytest.raises(ValueError, match="duplicate mapping key: 'kind'"):
    load_model_selection_policy(artifact)


def test_omitted_default_kind_is_rejected(tmp_path: Path) -> None:
  document = _selection_document()
  del document["capabilities"]["session.driver"]["default"]["kind"]

  with pytest.raises(ValueError, match="missing fields: kind"):
    load_model_selection_policy(_write_artifact(tmp_path / "selection.yaml", document))


def test_omitted_channel_default_kind_is_rejected(tmp_path: Path) -> None:
  document = _selection_document()
  document["capabilities"]["session.driver"]["by_channel"]["cli"] = {
    "model_key": "anthropic.claude-opus-5",
    "effort": "high",
  }

  with pytest.raises(ValueError, match="missing fields: kind"):
    load_model_selection_policy(_write_artifact(tmp_path / "selection.yaml", document))


# --- Adapter-declared support and startup closure (review items C2 and C3) ---
#
# Adapter support comes from the installed adapters' own declarations, never a
# hand-maintained table.  Startup closure runs over the CONFIGURED registry:
# every entry serving a gateway-executed capability must resolve to an
# installed declared adapter/profile/route at construction, while entries
# serving only externally-executed capabilities (risk.*, investment.*) are
# admitted registry facts for their own serving processes.


def _server_config(registry, policy) -> "GatewayServerConfig":
  async def _build_chat_runtime(session, request, channel, auth_manager):
    raise NotImplementedError("closure tests never run a chat turn")

  return GatewayServerConfig(
    jwt_secret="model-authority-closure-test-secret-01234",
    valid_api_keys={"closure-test-key"},
    tenant_id="gateway-tests",
    model_registry=registry,
    model_selection_policy=policy,
    build_chat_runtime=_build_chat_runtime,
  )


def test_hand_maintained_adapter_table_is_gone() -> None:
  import agent_gateway.model_registry as model_registry_module

  assert not hasattr(model_registry_module, "INITIAL_ADAPTER_ROUTE_SUPPORT")


def test_installed_declarations_cover_only_adapters_this_package_implements() -> None:
  supports = installed_adapter_route_support()

  assert set(supports) == {
    "anthropic.messages",
    "codex.responses",
    "openai.responses",
    "xai.responses",
  }
  # The Risk-local adapters are implemented in the Risk serving process; this
  # package must never vouch for them.
  assert "anthropic.sdk.messages" not in supports
  assert "openai.sdk.chat_completions" not in supports
  for adapter_id, support in supports.items():
    assert support.adapter == adapter_id
  assert supports["anthropic.messages"].provider == "anthropic"
  assert supports["codex.responses"].provider == "codex"
  assert supports["openai.responses"].provider == "openai"
  assert supports["xai.responses"].provider == "xai"


def test_packaged_gateway_executed_entries_are_declaration_supported() -> None:
  supports = installed_adapter_route_support()
  for entry in INITIAL_MODEL_REGISTRY.models.values():
    if not (set(entry.capabilities) & GATEWAY_EXECUTED_CAPABILITY_IDS):
      continue
    declaration = supports[entry.adapter]
    assert declaration.supports(entry), entry.key


def test_admission_requires_support_for_every_entry_without_designation(
  tmp_path: Path,
) -> None:
  # Fail-closed default: with no executed-capability designation, the packaged
  # registry (which carries externally-executed SDK entries) does NOT admit
  # against the installed declarations — exclusion only ever happens through
  # the explicit designation, never by silently skipping an entry.
  registry = load_model_registry(DEFAULT_MODEL_REGISTRY_ARTIFACT)

  with pytest.raises(ValueError, match="no installed adapter/profile/route support"):
    registry.admit_adapter_support(installed_adapter_route_support())


def test_admission_rejects_missing_adapter_for_executed_capability(
  tmp_path: Path,
) -> None:
  entry = dict(_NEW_MODEL_ENTRY)
  entry["adapter"] = "vendor.unknown"
  registry = load_model_registry(_registry_with_entry(tmp_path, entry))

  with pytest.raises(ValueError, match="no installed adapter/profile/route support"):
    registry.admit_adapter_support(
      installed_adapter_route_support(),
      executed_capability_ids=GATEWAY_EXECUTED_CAPABILITY_IDS,
    )


def test_server_construction_fails_loudly_on_missing_adapter(tmp_path: Path) -> None:
  entry = dict(_NEW_MODEL_ENTRY)
  entry["adapter"] = "vendor.unknown"
  registry = load_model_registry(_registry_with_entry(tmp_path, entry))
  policy = load_model_selection_policy(DEFAULT_MODEL_SELECTION_ARTIFACT)

  with pytest.raises(ValueError, match="no installed capability adapter"):
    create_gateway_app(_server_config(registry, policy))


def test_server_construction_fails_on_undeclared_protocol_profile(
  tmp_path: Path,
) -> None:
  # The C2 landmine, encoded: a chat-completions profile bound to the
  # Responses-only OpenAI adapter must fail construction — never boot green
  # and execute the wrong protocol.
  entry = dict(_NEW_MODEL_ENTRY)
  entry["provider"] = "openai"
  entry["upstream_model"] = "gpt-x-test"
  entry["adapter"] = "openai.responses"
  entry["protocol_profile"] = "chat_completions.standard"
  entry["route"] = "openai.public"
  entry["reported_identities"] = ["gpt-x-test"]
  registry = load_model_registry(_registry_with_entry(tmp_path, entry))
  policy = load_model_selection_policy(DEFAULT_MODEL_SELECTION_ARTIFACT)

  with pytest.raises(ValueError, match="does not declare"):
    create_gateway_app(_server_config(registry, policy))


def test_server_construction_admits_artifact_loaded_registry(tmp_path: Path) -> None:
  # The artifact-loaded closure path: an alternative admitted artifact (the
  # deployment-selection journey) constructs the server, with the
  # externally-executed SDK entries admitted as registry facts.
  registry = load_model_registry(_registry_with_entry(tmp_path, dict(_NEW_MODEL_ENTRY)))
  policy = load_model_selection_policy(_selection_allowing_new_driver(tmp_path))

  app = create_gateway_app(_server_config(registry, policy))

  assert app is not None


def test_import_time_closure_rejects_env_selected_registry_with_missing_adapter(
  tmp_path: Path,
) -> None:
  document = _registry_document()
  fable = next(
    entry for entry in document["models"] if entry["key"] == "anthropic.claude-fable-5"
  )
  fable["adapter"] = "vendor.unknown"
  artifact = _write_artifact(tmp_path / "registry.yaml", document)

  result = _run_module_import({MODEL_REGISTRY_FILE_ENV_VAR: str(artifact)})

  assert result.returncode != 0
  assert "no installed adapter/profile/route support" in result.stderr
