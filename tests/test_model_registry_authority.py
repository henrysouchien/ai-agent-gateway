from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.capability_binding import (
  AuthContext,
  CapabilityResolutionError,
  CredentialHandle,
  ModelSelectionIntent,
  eligible_model_choices,
  reauthorize_capability_bind,
  resolve_capability_model,
  validate_reported_identity,
)
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  CapabilityDefault,
  INITIAL_ADAPTER_ROUTE_SUPPORT,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
  ProductModelRegistry,
  ProductModelSelectionPolicy,
)


def _auth(
  *,
  providers: frozenset[str] = frozenset({"anthropic"}),
  capabilities: frozenset[str] = frozenset({"session.driver", "plan.author"}),
  model_keys: frozenset[str] = frozenset({"anthropic.claude-opus-5"}),
) -> AuthContext:
  handles = {
    provider: CredentialHandle(
      handle_id=f"credential:{provider}:user-7",
      provider=provider,
      principal="user",
      tenant_id="tenant-1",
      actor_id="user-7",
    )
    for provider in providers
  }
  return AuthContext(
    run_mode="interactive",
    actor_id="user-7",
    tenant_id="tenant-1",
    user_provider_handles=handles,
    service_provider_handles={},
    entitled_capabilities=capabilities,
    entitled_model_keys=model_keys,
  )


def _resolve_driver(
  auth: AuthContext,
  *,
  explicit_intent: ModelSelectionIntent | None = None,
  saved_preference: ModelSelectionIntent | None = None,
):
  return resolve_capability_model(
    "session.driver",
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth=auth,
    explicit_intent=explicit_intent,
    saved_preference=saved_preference,
  )


def test_initial_artifacts_are_complete_and_adapter_closed() -> None:
  assert set(INITIAL_MODEL_SELECTION_POLICY.capabilities) == CAPABILITY_IDS
  INITIAL_MODEL_SELECTION_POLICY.admit_registry(INITIAL_MODEL_REGISTRY)
  INITIAL_MODEL_REGISTRY.admit_adapter_support(INITIAL_ADAPTER_ROUTE_SUPPORT)
  assert "openai.gpt-4-1-sdk" not in INITIAL_MODEL_REGISTRY.models
  assert "openai.gpt-4o-mini" not in INITIAL_MODEL_REGISTRY.models
  assert "investment.benchmark_judge" not in CAPABILITY_IDS
  assert "investment.biotech_review" in CAPABILITY_IDS
  assert all(
    not policy.by_channel
    for policy in INITIAL_MODEL_SELECTION_POLICY.capabilities.values()
  )
  assert INITIAL_MODEL_SELECTION_POLICY.capabilities[
    "plan.author"
  ].allow_authenticated_run_override
  assert all(
    not policy.allow_authenticated_run_override
    for capability_id, policy in INITIAL_MODEL_SELECTION_POLICY.capabilities.items()
    if capability_id != "plan.author"
  )
  assert all(
    INITIAL_MODEL_SELECTION_POLICY.capabilities[capability_id].default.effort
    == "high"
    for capability_id in {
      "node.explore",
      "node.implement",
      "node.mutate",
      "node.verify",
      "node.choose",
    }
  )
  assert {
    entry.key
    for entry in INITIAL_MODEL_REGISTRY.models.values()
    if entry.capabilities.get("session.driver") == "user_selectable"
  } == {
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
  }
  assert INITIAL_MODEL_REGISTRY.require("openai.gpt-5-6").label == "GPT-5.6"
  assert INITIAL_MODEL_REGISTRY.require("codex.gpt-5-6-sol").label == "GPT-5.6 Sol"
  assert INITIAL_MODEL_SELECTION_POLICY.capabilities[
    "risk.asset_classification"
  ].default.model_key == "anthropic.claude-haiku-4-5-20251001-sdk"
  assert INITIAL_MODEL_SELECTION_POLICY.capabilities[
    "investment.newsletter"
  ].default.model_key == "anthropic.claude-haiku-4-5-20251001-gateway"
  assert all(
    entry.lifecycle == "hidden"
    for entry in INITIAL_MODEL_REGISTRY.models.values()
    if "user_selectable" not in entry.capabilities.values()
  )


def test_omitted_selection_resolves_complete_opus_five_high_bind() -> None:
  bind = _resolve_driver(_auth())

  assert bind.model_dump(mode="json") == {
    "schema_version": "1.0",
    "capability_id": "session.driver",
    "model_key": "anthropic.claude-opus-5",
    "provider": "anthropic",
    "upstream_model": "claude-opus-5",
    "adapter": "anthropic.messages",
    "protocol_profile": "messages.adaptive",
    "route": "anthropic.public",
    "effort": "high",
    "credential_principal": "user",
    "credential_ref": "credential:anthropic:user-7",
    "run_mode": "interactive",
    "registry_revision": "2026-08-13.1",
    "policy_revision": "2026-08-13.1",
    "selection_source": "capability_default",
  }


def test_explicit_stable_key_is_honored_exactly() -> None:
  auth = _auth(
    providers=frozenset({"anthropic", "xai"}),
    model_keys=frozenset({"anthropic.claude-opus-5", "xai.grok-4-5"}),
  )
  bind = _resolve_driver(
    auth,
    explicit_intent=ModelSelectionIntent(
      model_key="xai.grok-4-5",
      effort="low",
      source="explicit_user",
      catalog_revision="observed-old-revision",
    ),
  )

  assert bind.model_key == "xai.grok-4-5"
  assert bind.provider == "xai"
  assert bind.upstream_model == "grok-4.5"
  assert bind.effort == "low"
  assert bind.selection_source == "explicit_user"


def test_unknown_explicit_key_is_rejected_without_substitution() -> None:
  with pytest.raises(CapabilityResolutionError) as caught:
    _resolve_driver(
      _auth(),
      explicit_intent=ModelSelectionIntent(
        model_key="anthropic.not-a-model",
        effort=None,
        source="explicit_user",
      ),
    )

  assert caught.value.code == "capability_model_unavailable"
  assert caught.value.model_key == "anthropic.not-a-model"


def test_stale_saved_preference_falls_back_to_default_but_is_not_rewritten() -> None:
  preference = ModelSelectionIntent(
    model_key="xai.grok-4-5",
    effort="low",
    source="saved_preference",
  )

  bind = _resolve_driver(_auth(), saved_preference=preference)

  assert bind.model_key == "anthropic.claude-opus-5"
  assert bind.selection_source == "capability_default"
  assert preference.model_key == "xai.grok-4-5"


def test_ineligible_default_requires_action_instead_of_first_available() -> None:
  auth = _auth(
    providers=frozenset({"xai"}),
    model_keys=frozenset({"xai.grok-4-5"}),
  )

  with pytest.raises(CapabilityResolutionError) as caught:
    _resolve_driver(auth)

  assert caught.value.code == "default_not_eligible"
  assert caught.value.model_key == "anthropic.claude-opus-5"
  assert caught.value.eligible_model_keys == ("xai.grok-4-5",)


def test_plan_author_inherits_exact_parent_by_default() -> None:
  auth = _auth()
  parent = _resolve_driver(auth)

  author = resolve_capability_model(
    "plan.author",
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth=auth,
    parent_bind=parent,
  )

  assert author.model_key == parent.model_key
  assert author.provider == parent.provider
  assert author.upstream_model == parent.upstream_model
  assert author.adapter == parent.adapter
  assert author.protocol_profile == parent.protocol_profile
  assert author.route == parent.route
  assert author.effort == parent.effort
  assert author.credential_ref == parent.credential_ref
  assert author.selection_source == "parent_binding"


def test_node_fork_can_inherit_any_eligible_session_driver_binding() -> None:
  auth = _auth(
    providers=frozenset({"xai"}),
    capabilities=frozenset({"session.driver", "node.fork"}),
    model_keys=frozenset({"xai.grok-4-5"}),
  )
  parent = _resolve_driver(
    auth,
    explicit_intent=ModelSelectionIntent(
      model_key="xai.grok-4-5",
      effort="high",
      source="explicit_user",
    ),
  )

  fork = resolve_capability_model(
    "node.fork",
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth=auth,
    parent_bind=parent,
  )

  assert fork.model_key == parent.model_key
  assert fork.effort == parent.effort
  assert fork.selection_source == "parent_binding"


def test_durable_bind_reauthorization_never_reselects() -> None:
  original_auth = _auth()
  bind = _resolve_driver(original_auth).model_copy(
    update={"registry_revision": "older", "policy_revision": "older"}
  )
  reauthorized = reauthorize_capability_bind(
    bind,
    registry=INITIAL_MODEL_REGISTRY,
    auth=original_auth,
  )
  assert reauthorized.handle_id == bind.credential_ref

  alternative_only = _auth(
    providers=frozenset({"xai"}),
    model_keys=frozenset({"anthropic.claude-opus-5", "xai.grok-4-5"}),
  )
  with pytest.raises(CapabilityResolutionError) as caught:
    reauthorize_capability_bind(
      bind,
      registry=INITIAL_MODEL_REGISTRY,
      auth=alternative_only,
    )
  assert caught.value.code == "credential_unavailable"
  assert caught.value.model_key == bind.model_key


def test_durable_reauthorization_ignores_current_selection_policy_change() -> None:
  auth = _auth(
    model_keys=frozenset({
      "anthropic.claude-opus-5",
      "anthropic.claude-sonnet-5",
    }),
  )
  bind = _resolve_driver(auth)
  changed_capabilities = dict(INITIAL_MODEL_SELECTION_POLICY.capabilities)
  changed_capabilities["session.driver"] = replace(
    changed_capabilities["session.driver"],
    default=CapabilityDefault(
      kind="model",
      model_key="anthropic.claude-sonnet-5",
      effort="high",
    ),
    allowed_model_keys=frozenset({"anthropic.claude-sonnet-5"}),
  )
  changed_policy = ProductModelSelectionPolicy(
    schema="product-model-selection/v1",
    revision="later-policy",
    capabilities=changed_capabilities,
  )
  changed_policy.admit_registry(INITIAL_MODEL_REGISTRY)
  new_bind = resolve_capability_model(
    "session.driver",
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=changed_policy,
    auth=auth,
  )

  handle = reauthorize_capability_bind(
    bind,
    registry=INITIAL_MODEL_REGISTRY,
    auth=auth,
  )

  assert handle.handle_id == bind.credential_ref
  assert bind.model_key == "anthropic.claude-opus-5"
  assert new_bind.model_key == "anthropic.claude-sonnet-5"


@pytest.mark.parametrize(
  ("lifecycle", "error_code"),
  [
    ("deprecated", None),
    ("disabled", "capability_model_disabled"),
    ("revoked", "capability_model_revoked"),
  ],
)
def test_durable_reauthorization_applies_current_lifecycle_only(
  lifecycle: str,
  error_code: str | None,
) -> None:
  auth = _auth()
  bind = _resolve_driver(auth)
  entries = dict(INITIAL_MODEL_REGISTRY.models)
  entries[bind.model_key] = replace(
    entries[bind.model_key],
    lifecycle=lifecycle,
    capabilities={
      capability_id: "internal"
      for capability_id in entries[bind.model_key].capabilities
    },
  )
  current_registry = ProductModelRegistry(
    schema="product-model-registry/v1",
    revision=f"lifecycle-{lifecycle}",
    models=entries,
  )

  if error_code is None:
    assert reauthorize_capability_bind(
      bind,
      registry=current_registry,
      auth=auth,
    ).handle_id == bind.credential_ref
    return

  with pytest.raises(CapabilityResolutionError) as caught:
    reauthorize_capability_bind(
      bind,
      registry=current_registry,
      auth=auth,
    )
  assert caught.value.code == error_code


def test_durable_reauthorization_blocks_removed_capability_qualification() -> None:
  auth = _auth()
  bind = _resolve_driver(auth)
  entries = dict(INITIAL_MODEL_REGISTRY.models)
  entry = entries[bind.model_key]
  entries[bind.model_key] = replace(
    entry,
    capabilities={
      capability_id: exposure
      for capability_id, exposure in entry.capabilities.items()
      if capability_id != bind.capability_id
    },
  )
  current_registry = ProductModelRegistry(
    schema="product-model-registry/v1",
    revision="qualification-removed",
    models=entries,
  )

  with pytest.raises(CapabilityResolutionError) as caught:
    reauthorize_capability_bind(
      bind,
      registry=current_registry,
      auth=auth,
    )

  assert caught.value.code == "capability_model_not_allowed"


def test_eligible_choices_are_authenticated_and_provider_free() -> None:
  choices = eligible_model_choices(
    "session.driver",
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth=_auth(),
  )

  assert [choice.key for choice in choices] == ["anthropic.claude-opus-5"]
  assert not hasattr(choices[0], "provider")
  assert not hasattr(choices[0], "upstream_model")


def test_reported_identity_is_recorded_verbatim_or_rejected() -> None:
  bind = _resolve_driver(_auth())
  assert validate_reported_identity(
    bind,
    "claude-opus-5",
    registry=INITIAL_MODEL_REGISTRY,
  ) == "claude-opus-5"

  with pytest.raises(CapabilityResolutionError) as caught:
    validate_reported_identity(
      bind,
      "claude-opus-5-unreviewed-snapshot",
      registry=INITIAL_MODEL_REGISTRY,
    )
  assert caught.value.code == "reported_identity_mismatch"


def test_interactive_haiku_admits_the_provider_dated_report_without_rebinding() -> None:
  bind = _resolve_driver(
    _auth(model_keys=frozenset({"anthropic.claude-haiku-4-5"})),
    explicit_intent=ModelSelectionIntent(
      model_key="anthropic.claude-haiku-4-5",
      effort="none",
      source="explicit_user",
    ),
  )

  assert bind.model_key == "anthropic.claude-haiku-4-5"
  assert bind.upstream_model == "claude-haiku-4-5"
  assert validate_reported_identity(
    bind,
    "claude-haiku-4-5-20251001",
    registry=INITIAL_MODEL_REGISTRY,
  ) == "claude-haiku-4-5-20251001"

  with pytest.raises(CapabilityResolutionError) as caught:
    validate_reported_identity(
      bind,
      "claude-haiku-4-5-20990101",
      registry=INITIAL_MODEL_REGISTRY,
    )
  assert caught.value.code == "reported_identity_mismatch"
